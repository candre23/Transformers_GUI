#!/usr/bin/env python3
"""
Transformers GUI — a very basic Python GUI to load and chat with HF Transformers models,
optionally quantized via bitsandbytes (4-bit/8-bit).

• Pick CUDA or CPU
• Select GPU(s) to use (requires restart to take effect)
• Choose full precision, 8-bit, or 4-bit quantization
• Load from local folder OR directly from a Hugging Face repo (with a chosen download dir)
• Simple chat window with streaming outputs

Notes
-----
- bitsandbytes must be installed with CUDA support for 8-bit/4-bit.
- Some newer models require `trust_remote_code=True` to load custom code. A toggle is provided in the GUI.
- If the tokenizer defines a chat template, it will be used via `tokenizer.apply_chat_template`.
  Otherwise, a simple fallback prompt format is used.

Dependencies (install examples)
-------------------------------
python -m pip install --upgrade "transformers>=4.44" accelerate torch
python -m pip install bitsandbytes huggingface_hub

If on Windows with WSL2 + NVIDIA, ensure a matching CUDA-enabled PyTorch build.

Copyleft 2025 - no rights reserved.  Do whatever you want with this vibeslop.
"""

import os
import sys
import threading
import time
import gc
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Transformers / HF
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

try:
    import bitsandbytes as bnb  # noqa: F401
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False

# Optional Flash-Attention 2 (for much faster attention on CUDA)
try:
    import flash_attn  # noqa: F401
    FA_AVAILABLE = True
except Exception:
    FA_AVAILABLE = False

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


# Quiet down noisy advisory warnings (optional & safe)
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

import logging
try:
    logging.getLogger("xielu").setLevel(logging.ERROR)
except Exception:
    pass

class _EventStoppingCriteria(StoppingCriteria):
    """Stop generation when an external threading.Event is set."""
    def __init__(self, event):
        super().__init__()
        self._event = event

    # works across transformers versions that pass (input_ids, scores, **kw)
    def __call__(self, input_ids, scores, **kwargs):
        return bool(self._event.is_set())


class LLMLabGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Transformers GUI")
        self.geometry("980x720")

        # State
        self.model = None
        self.tokenizer = None
        self.stop_generation = threading.Event()
        self.generation_thread = None
        self._gen_worker = None
        self._current_streamer = None

        self.chat_messages = []  # list of {"role": "system"|"user"|"assistant", "content": str}

        # Notebook with two tabs
        self.nb = ttk.Notebook(self)
        self.nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.tab_model = ttk.Frame(self.nb)
        self.tab_chat = ttk.Frame(self.nb)
        self.nb.add(self.tab_model, text="Model")
        self.nb.add(self.tab_chat, text="Chat")

        # --- Top controls frame
        top = ttk.Frame(self.tab_model, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        # Compute
        ttk.Label(top, text="Compute:").grid(row=0, column=0, sticky=tk.W)
        self.compute_var = tk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")
        self.compute_combo = ttk.Combobox(top, textvariable=self.compute_var, values=["cuda", "cpu"], width=7, state="readonly")
        self.compute_combo.grid(row=0, column=1, padx=6)
        if not torch.cuda.is_available():
            self.compute_combo.configure(state="disabled")
            ttk.Label(top, text="(CUDA not detected)", foreground="#a00").grid(row=0, column=2, sticky=tk.W)

        # Quantization
        ttk.Label(top, text="Quantization:").grid(row=0, column=3, sticky=tk.W, padx=(16, 0))
        self.quant_var = tk.StringVar(value="4bit")
        quant_values = ["full", "8bit", "4bit"]
        self.quant_combo = ttk.Combobox(top, textvariable=self.quant_var, values=quant_values, width=7, state="readonly")
        self.quant_combo.grid(row=0, column=4, padx=6)
        if not BNB_AVAILABLE:
            self.quant_combo.configure(state="disabled")
            ttk.Label(top, text="(bitsandbytes not installed)", foreground="#a00").grid(row=0, column=5, sticky=tk.W)

        # Trust remote code
        self.trust_remote_var = tk.BooleanVar(value=True)
        self.trust_remote_chk = ttk.Checkbutton(top, text="trust_remote_code", variable=self.trust_remote_var)
        self.trust_remote_chk.grid(row=0, column=6, padx=(16, 0))
        # Flash-Attn toggle
        self.flash_var = tk.BooleanVar(value=False)
        self.flash_chk = ttk.Checkbutton(top, text="Flash-Attn 2", variable=self.flash_var)
        self.flash_chk.grid(row=0, column=7, padx=(16, 0))

        # Reasoning mode
        self.reason_var = tk.BooleanVar(value=False)
        self.reason_chk = ttk.Checkbutton(top, text="Reasoning mode (final answers only)", variable=self.reason_var)
        self.reason_chk.grid(row=0, column=8, padx=(16, 0))

        # --- Optional GPU selection (for CUDA). Lets you pick which GPUs are visible to the loader.
        self._original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.gpu_vars = []
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            gpu_box = ttk.Labelframe(self.tab_model, text="CUDA GPUs to use (applies on next Load)", padding=6)
            gpu_box.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))
            cols = 3
            for i in range(torch.cuda.device_count()):
                try:
                    name = torch.cuda.get_device_name(i)
                except Exception:
                    name = f"GPU {i}"
                var = tk.IntVar(value=1)
                self.gpu_vars.append(var)
                r, c = divmod(i, cols)
                ttk.Checkbutton(gpu_box, text=f"[{i}] {name}", variable=var).grid(row=r, column=c, sticky=tk.W, padx=6, pady=2)

            # quick select buttons
            def _all_on():
                for v in self.gpu_vars: v.set(1)
            def _all_off():
                for v in self.gpu_vars: v.set(0)
            ttk.Button(gpu_box, text="All", command=_all_on).grid(row=99, column=0, sticky=tk.W, pady=(4,0))
            ttk.Button(gpu_box, text="None", command=_all_off).grid(row=99, column=1, sticky=tk.W, pady=(4,0))

            # Restart to guarantee CUDA visibility takes effect before enumeration
            def _restart_with_selected():
                selected = [i for i, v in enumerate(self.gpu_vars) if v.get() == 1]
                if not selected:
                    messagebox.showwarning("No GPU selected", "Select at least one GPU or switch compute to CPU.")
                    return
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in selected)
                messagebox.showinfo("Restarting", f"Restarting with CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
                os.execve(sys.executable, [sys.executable] + sys.argv, env)

            ttk.Button(gpu_box, text="Restart with selected GPUs", command=_restart_with_selected).grid(row=99, column=2, sticky=tk.W, pady=(4,0))

        # --- Model source
        source = ttk.Labelframe(self.tab_model, text="Model Source", padding=8)
        source.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.source_var = tk.StringVar(value="hf")
        ttk.Radiobutton(source, text="Hugging Face repo", variable=self.source_var, value="hf").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(source, text="Local folder", variable=self.source_var, value="local").grid(row=0, column=1, sticky=tk.W, padx=(12, 0))

        # HF repo id
        ttk.Label(source, text="HF repo id (e.g. Qwen/Qwen3-8B):").grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))
        self.repo_var = tk.StringVar(value="")
        ttk.Entry(source, textvariable=self.repo_var, width=60).grid(row=2, column=0, columnspan=4, sticky=tk.W)

        # Download dir
        ttk.Label(source, text="Download / cache directory:").grid(row=3, column=0, sticky=tk.W, pady=(6, 0))
        self.download_dir_var = tk.StringVar(value=os.path.expanduser("~/.cache/huggingface/models"))
        dl_entry = ttk.Entry(source, textvariable=self.download_dir_var, width=50)
        dl_entry.grid(row=3, column=1, sticky=tk.W)
        ttk.Button(source, text="Browse…", command=self._pick_download_dir).grid(row=3, column=2, padx=6)

        # Local dir
        ttk.Label(source, text="Local model folder:").grid(row=4, column=0, sticky=tk.W, pady=(6, 0))
        self.local_dir_var = tk.StringVar(value="")
        local_entry = ttk.Entry(source, textvariable=self.local_dir_var, width=50)
        local_entry.grid(row=4, column=1, sticky=tk.W)
        ttk.Button(source, text="Browse…", command=self._pick_local_dir).grid(row=4, column=2, padx=6)

        # --- Generation params
        params = ttk.Labelframe(self.tab_chat, text="Generation Settings", padding=8)
        params.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(params, text="max_new_tokens:").grid(row=0, column=0, sticky=tk.W)
        self.max_new_tokens = tk.IntVar(value=512)
        ttk.Spinbox(params, from_=1, to=8192, textvariable=self.max_new_tokens, width=6).grid(row=0, column=1, padx=6)

        ttk.Label(params, text="temperature:").grid(row=0, column=2, sticky=tk.W, padx=(16, 0))
        self.temperature = tk.DoubleVar(value=0.7)
        ttk.Spinbox(params, from_=0.0, to=2.0, increment=0.05, textvariable=self.temperature, width=6).grid(row=0, column=3, padx=6)

        ttk.Label(params, text="top_p:").grid(row=0, column=4, sticky=tk.W, padx=(16, 0))
        self.top_p = tk.DoubleVar(value=0.95)
        ttk.Spinbox(params, from_=0.0, to=1.0, increment=0.01, textvariable=self.top_p, width=6).grid(row=0, column=5, padx=6)

        # System instruction (optional)
        ttk.Label(params, text="System instruction (optional):").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        self.system_text = tk.Text(params, height=2, width=90)
        self.system_text.grid(row=2, column=0, columnspan=6, sticky=tk.W)

        # More sampling controls
        row = 3
        ttk.Label(params, text="top_k:").grid(row=row, column=0, sticky=tk.W, pady=(8,0))
        self.top_k = tk.IntVar(value=50)
        ttk.Spinbox(params, from_=0, to=2048, textvariable=self.top_k, width=6).grid(row=row, column=1, padx=6, pady=(8,0))

        ttk.Label(params, text="repetition_penalty:").grid(row=row, column=2, sticky=tk.W, padx=(16,0), pady=(8,0))
        self.repetition_penalty = tk.DoubleVar(value=1.05)
        ttk.Spinbox(params, from_=0.5, to=2.0, increment=0.01, textvariable=self.repetition_penalty, width=6).grid(row=row, column=3, padx=6, pady=(8,0))

        ttk.Label(params, text="typical_p:").grid(row=row, column=4, sticky=tk.W, padx=(16,0), pady=(8,0))
        self.typical_p = tk.DoubleVar(value=1.0)
        ttk.Spinbox(params, from_=0.0, to=1.0, increment=0.01, textvariable=self.typical_p, width=6).grid(row=row, column=5, padx=6, pady=(8,0))

        row += 1
        ttk.Label(params, text="no_repeat_ngram_size:").grid(row=row, column=0, sticky=tk.W)
        self.no_repeat_ngram = tk.IntVar(value=0)
        ttk.Spinbox(params, from_=0, to=20, textvariable=self.no_repeat_ngram, width=6).grid(row=row, column=1, padx=6)

        # Load / unload
        load_frame = ttk.Frame(self.tab_model, padding=(8, 0))
        load_frame.pack(side=tk.TOP, fill=tk.X)
        self.load_btn = ttk.Button(load_frame, text="Load model", command=self.load_model)
        self.load_btn.pack(side=tk.LEFT)
        ttk.Button(load_frame, text="Unload", command=self.unload_model).pack(side=tk.LEFT, padx=8)

        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(load_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=16)
        # --- Chat area (grid-based so the input + buttons are always visible)
        chat = ttk.Labelframe(self.tab_chat, text="Chat", padding=8)
        chat.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Make the chat area stretchy
        chat.grid_rowconfigure(0, weight=1)
        chat.grid_columnconfigure(0, weight=1)  # main text column
        chat.grid_columnconfigure(1, weight=0)  # scrollbar column
        chat.grid_columnconfigure(2, weight=0)  # buttons

        # Transcript (with scrollbar)
        self.chat_view = tk.Text(chat, wrap=tk.WORD, state=tk.DISABLED)
        chat_scroll = ttk.Scrollbar(chat, orient="vertical", command=self.chat_view.yview)
        self.chat_view.configure(yscrollcommand=chat_scroll.set)
        self.chat_view.grid(row=0, column=0, sticky="nsew")
        chat_scroll.grid(row=0, column=1, sticky="ns")

        # Input box (taller, with scrollbar)
        self.user_entry = tk.Text(chat, height=6, wrap=tk.WORD)
        input_scroll = ttk.Scrollbar(chat, orient="vertical", command=self.user_entry.yview)
        self.user_entry.configure(yscrollcommand=input_scroll.set)
        self.user_entry.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        input_scroll.grid(row=1, column=1, sticky="nsw", pady=(6, 0))

        # Buttons (right column of the Chat grid)
        self.reset_btn = ttk.Button(chat, text="Reset chat", command=self.reset_chat, width=14)
        self.reset_btn.grid(row=0, column=2, sticky="ne", padx=(6, 0), pady=(0, 0))

        self.send_btn = ttk.Button(chat, text="Send", command=self.on_send, width=14)
        self.send_btn.grid(row=1, column=2, sticky="e", padx=(6, 0), pady=(6, 0))

        self.stop_btn = ttk.Button(chat, text="Stop", command=self.on_stop, state=tk.DISABLED, width=14)
        self.stop_btn.grid(row=2, column=2, sticky="e", padx=(6, 0), pady=(6, 8))

        # Diagnostics line (tokens/sec)
        self.diag_var = tk.StringVar(value="t/s: —")
        ttk.Label(chat, textvariable=self.diag_var).grid(row=2, column=0, sticky="w", pady=(6, 8))

        # Keyboard shortcut: Ctrl+Enter to send, Shift+Enter for newline
        def _send_shortcut(event):
            self.on_send(); return "break"
        self.user_entry.bind("<Control-Return>", _send_shortcut)

        # ----------------- UI helpers
    def _pick_download_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.download_dir_var.set(d)

    def _pick_local_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.local_dir_var.set(d)

    def _clear_conversation_state(self):
        """
        Best-effort wipe of any conversation buffers we keep between turns.
        Transformers models are stateless; we store history locally. This
        clears those local structures so a new chat starts clean.
        """
        # your transcript/messages list
        try:
            self.chat_messages.clear()
        except Exception:
            pass

        # clear any book-keeping we might use for streaming/threads
        for attr in ("_current_streamer", "_gen_worker"):
            if hasattr(self, attr):
                setattr(self, attr, None)

        # if you keep any cache-like attrs, clear them (harmless if absent)
        for attr in (
            "past_key_values",
            "_past_key_values",
            "_past",
            "chat_history_ids",
            "conversation_tokens",
            "_kv_cache",
            "cache_position",
        ):
            if hasattr(self, attr):
                setattr(self, attr, None)


    def reset_chat(self):
        """
        Stop generation (if running) and reset the conversation/UI to a blank state.
        Keeps the loaded model and the System Instruction field as-is.
        """
        # 1) stop any in-flight generation and let the worker unwind quickly
        try:
            self.on_stop()
        except Exception:
            pass
        try:
            # allow future runs to proceed
            self.stop_generation.clear()
        except Exception:
            pass

        # 2) clear local conversation state
        self._clear_conversation_state()

        # 3) clear the UI transcript and input box
        try:
            self.chat_view.configure(state=tk.NORMAL)
            self.chat_view.delete("1.0", tk.END)
            self.chat_view.configure(state=tk.DISABLED)
        except Exception:
            pass
        try:
            self.user_entry.delete("1.0", tk.END)
        except Exception:
            pass

        # 4) reset diagnostics
        try:
            self._set_diag("t/s: —")
            self._set_status("Chat reset.")
        except Exception:
            pass
            
        # Reset buttons?  Hopefully?
        try:
            self.send_btn.configure(state=tk.NORMAL)
            self.stop_btn.configure(state=tk.DISABLED)
        except Exception:
            pass


    def _append_chat(self, role: str, text: str):
        """Append a message to the transcript with blank lines between turns."""
        self.chat_messages.append({"role": role, "content": text})
        self.chat_view.configure(state=tk.NORMAL)
        # Insert a blank line if there's already content
        existing = self.chat_view.get("1.0", tk.END)
        if existing and existing.strip():
            self.chat_view.insert(tk.END, "\n\n")


        # Write the role and content, followed by a blank line
        line = f"{role.capitalize()}: {text}\n\n"
        self.chat_view.insert(tk.END, line)
        self.chat_view.configure(state=tk.DISABLED)
        self.chat_view.see(tk.END)

    def _append_stream_text(self, text: str):
        self.chat_view.configure(state=tk.NORMAL)
        self.chat_view.insert(tk.END, text)
        self.chat_view.configure(state=tk.DISABLED)
        self.chat_view.see(tk.END)

    def _set_status(self, s: str):
        self.status_var.set(s)
        self.update_idletasks()

    def _set_diag(self, s: str):
        try:
            self.diag_var.set(s)
            self.update_idletasks()
        except Exception:
            pass

    # ----------------- Model loading/unloading
    def load_model(self):
        if self.model is not None:
            messagebox.showinfo("Model already loaded", "Please unload first or use the current model.")
            return

        source = self.source_var.get()
        compute = self.compute_var.get()
        quant = self.quant_var.get() if BNB_AVAILABLE else "full"
        trust_remote = self.trust_remote_var.get()

        # Apply CUDA GPU selection before we touch the model if using CUDA
        if compute == "cuda" and torch.cuda.is_available() and self.gpu_vars:
            selected = [i for i, v in enumerate(self.gpu_vars) if v.get() == 1]
            if not selected:
                messagebox.showwarning("No GPU selected", "Please select at least one GPU or switch compute to CPU.")
                return
            # Best-effort: limit visible devices so Accelerate shards only across these
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in selected)
            self._set_status(f"Using CUDA devices: {selected}")

        repo_id = self.repo_var.get().strip()
        download_dir = self.download_dir_var.get().strip()
        local_dir = self.local_dir_var.get().strip()

        # Resolve model path
        if source == "hf":
            if not HF_AVAILABLE:
                messagebox.showerror("huggingface_hub not installed", "Install huggingface_hub to pull from HF repos.")
                return
            if not repo_id:
                messagebox.showerror("Missing repo id", "Please enter a Hugging Face repo id.")
                return
            # Download to specified dir (or reuse cache)
            self._set_status(f"Downloading {repo_id} (or using cache)…")
            try:
                model_path = snapshot_download(repo_id, cache_dir=download_dir, local_dir=None)
            except Exception as e:
                messagebox.showerror("HF download failed", str(e))
                self._set_status("Idle.")
                return
        else:
            model_path = local_dir
            if not model_path or not os.path.isdir(model_path):
                messagebox.showerror("Invalid local folder", "Please choose a valid local model folder.")
                return

        # Build kwargs
        model_kwargs = {}
        tokenizer_kwargs = {"use_fast": True}

        if compute == "cuda" and torch.cuda.is_available():
            device_map = "auto"  # let Accelerate shard across GPUs
            # Attention backend selection
            if self.flash_var.get():
                if FA_AVAILABLE:
                    model_kwargs.update(dict(attn_implementation="flash_attention_2"))
                else:
                    messagebox.showwarning("Flash-Attn unavailable", "flash-attn not installed or incompatible. Falling back to default attention.")
            if quant == "4bit":
                if not BNB_AVAILABLE:
                    messagebox.showerror("Missing bitsandbytes", "Install bitsandbytes for 4-bit loading.")
                    return
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs.update(
                    dict(
                        quantization_config=bnb_config,
                        device_map=device_map,
                        dtype=torch.bfloat16,
                    )
                )
            elif quant == "8bit":
                if not BNB_AVAILABLE:
                    messagebox.showerror("Missing bitsandbytes", "Install bitsandbytes for 8-bit loading.")
                    return
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs.update(
                    dict(
                        quantization_config=bnb_config,
                        device_map=device_map,
                        dtype=torch.bfloat16,
                    )
                )
            else:  # full on CUDA
                model_kwargs.update(
                    dict(
                        device_map=device_map,
                        dtype=torch.bfloat16,
                    )
                )
        else:
            # CPU path: WARNING for large models
            if quant != "full":
                messagebox.showwarning("Quantization on CPU", "8-bit/4-bit via bitsandbytes is generally for CUDA. Falling back to full precision on CPU.")
            model_kwargs.update(dict(device_map="cpu", dtype=torch.float32))

        # Common flags
        model_kwargs.update(dict(trust_remote_code=trust_remote))

        # Load tokenizer/model
        self._set_status(f"Loading tokenizer from {model_path}…")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        except Exception as e:
            messagebox.showerror("Tokenizer load failed", str(e))
            self._set_status("Idle.")
            return

        self._set_status(f"Loading model from {model_path}… (this can take a while)")
        self.load_btn.configure(state=tk.DISABLED)
        self.update_idletasks()

        def _load():
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
                # Some models benefit from config tweaks
                try:
                    self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                except Exception:
                    pass
                self._set_status("Model loaded.")
                # Diagnostics: which attention backend and visible GPUs
                try:
                    attn_impl = getattr(getattr(self.model, "config", None), "_attn_implementation", None) or \
                                getattr(getattr(self.model, "config", None), "attn_implementation", None) or "default"
                except Exception:
                    attn_impl = "unknown"
                try:
                    vis = []
                    for i in range(torch.cuda.device_count()):
                        vis.append(f"[{i}] {torch.cuda.get_device_name(i)}")
                    vis_str = ", ".join(vis) if vis else "(none)"
                except Exception:
                    vis_str = "(n/a)"
                self._set_diag(f"attention: {attn_impl}; visible CUDA: {vis_str}")
                messagebox.showinfo("Loaded", "Model is ready.")
            except Exception as e:
                self.model = None
                self._set_status("Idle.")
                messagebox.showerror("Model load failed", str(e))
            finally:
                self.load_btn.configure(state=tk.NORMAL)

        threading.Thread(target=_load, daemon=True).start()

    def unload_model(self):
        self.on_stop()
        self.model = None
        self.tokenizer = None
        self.chat_messages.clear()
        # restore original CUDA visibility if we changed it
        if getattr(self, "_original_cuda_visible", None) is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._original_cuda_visible
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        self._set_status("Unloaded.")
        messagebox.showinfo("Unloaded", "Model and tokenizer cleared from memory.")

    # ----------------- Chat / Generation
    def on_send(self):
        if self.model is None or self.tokenizer is None:
            messagebox.showwarning("No model", "Load a model first.")
            return

        user_text = self.user_entry.get("1.0", tk.END).strip()
        if not user_text:
            return

        # Append system message(s) if first turn
        if not any(m["role"] == "system" for m in self.chat_messages):
            sys_from_box = self.system_text.get("1.0", tk.END).strip()
            if sys_from_box:
                self.chat_messages.append({"role": "system", "content": sys_from_box})
            if self.reason_var.get():
                # Encourage private reasoning but final answer only
                self.chat_messages.append({
                    "role": "system",
                    "content": "You may do any private reasoning needed, but return only the final answer in clear prose. Do not include your intermediate reasoning or steps."
                })

        self._append_chat("user", user_text)
        self.user_entry.delete("1.0", tk.END)

        # Prepare prompt
        messages = self.chat_messages.copy()

        # Start streaming generation on a background thread
        self.stop_generation.clear()
        self.send_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)

        def _generate_thread():
            try:
                prompt_text = None
                if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                    prompt_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    # Simple fallback format
                    parts = []
                    for m in messages:
                        if m["role"] == "system":
                            parts.append(f"[SYSTEM] {m['content']}")
                        elif m["role"] == "user":
                            parts.append(f"User: {m['content']}")
                        elif m["role"] == "assistant":
                            parts.append(f"Assistant: {m['content']}")
                    parts.append("Assistant:")
                    prompt_text = "\n".join(parts)

                inputs = self.tokenizer(prompt_text, return_tensors="pt")

                if next(self.model.parameters()).is_cuda:
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                self._current_streamer = streamer

                gen_kwargs = dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=self.max_new_tokens.get(),
                    do_sample=True,
                    temperature=float(self.temperature.get()),
                    top_p=float(self.top_p.get()),
                    top_k=int(self.top_k.get()),
                    repetition_penalty=float(self.repetition_penalty.get()),
                    typical_p=float(self.typical_p.get()),
                    no_repeat_ngram_size=int(self.no_repeat_ngram.get()),
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                # Start generation in a worker thread for the streamer
                def _worker():
                    try:
                        self.model.generate(
                            **gen_kwargs,
                            stopping_criteria=StoppingCriteriaList([
                                _EventStoppingCriteria(self.stop_generation)
                            ])
                        )
                    except Exception as e:
                        self._append_stream_text("\n[Generation error: {}]\n".format(e))
                    finally:
                        pass


                t = threading.Thread(target=_worker, daemon=True)
                t.start()
                self._gen_worker = t

                # Read tokens
                self._append_chat("assistant", "")  # open an assistant block

                accum = []
                start_time = None
                last_diag = 0.0
                for token in streamer:
                    if self.stop_generation.is_set():
                        break
                    if start_time is None:
                        start_time = time.time()
                    accum.append(token)
                    self._append_stream_text(token)

                    # Update t/s roughly every 0.5s
                    now = time.time()
                    if start_time and (now - last_diag) >= 0.5:
                        gen_text = "".join(accum)
                        try:
                            gen_tokens = len(self.tokenizer(gen_text, add_special_tokens=False).input_ids)
                        except Exception:
                            gen_tokens = max(1, len(gen_text) // 3)
                        elapsed = now - start_time
                        tps = gen_tokens / max(elapsed, 1e-6)
                        self._set_diag(f"generated: {gen_tokens} tok in {elapsed:.1f}s — {tps:.2f} t/s")
                        last_diag = now

                # Save final assistant message content back to history
                final_text = "".join(accum)
                self.chat_messages.append({"role": "assistant", "content": final_text})

                # final diagnostics
                if start_time is not None:
                    elapsed = time.time() - start_time
                    try:
                        gen_tokens = len(self.tokenizer(final_text, add_special_tokens=False).input_ids)
                    except Exception:
                        gen_tokens = max(1, len(final_text) // 3)
                    tps = gen_tokens / max(elapsed, 1e-6)
                    self._set_diag(f"generated: {gen_tokens} tok in {elapsed:.1f}s — {tps:.2f} t/s")

            except Exception as e:
                self._append_stream_text("\n[Error: {}]\n".format(e))

            finally:
                self.send_btn.configure(state=tk.NORMAL)
                self.stop_btn.configure(state=tk.DISABLED)
                self._current_streamer = None
                self._gen_worker = None


        self.generation_thread = threading.Thread(target=_generate_thread, daemon=True)
        self.generation_thread.start()


    def _flush_all_cuda_caches(self):
        """Best-effort: flush allocator & IPC handles on every visible GPU."""
        try:
            if not torch.cuda.is_available():
                return
            n = torch.cuda.device_count()
            for i in range(n):
                try:
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                except Exception:
                    pass
            try:
                torch.cuda.set_device(0)  # harmless if it fails
            except Exception:
                pass
        except Exception:
            pass


    def unload_model(self):
        """
        Safe, in-place unload: stop gen, drop refs, and free VRAM across all GPUs.
        Leaves the app running.
        """
        # Stop any in-flight generation
        try:
            self.on_stop()
            t = getattr(self, "generation_thread", None)
            if t and t.is_alive():
                t.join(timeout=3.0)
        except Exception:
            pass

        # Move weights off-GPU (best-effort), then drop refs
        try:
            if self.model is not None:
                with torch.no_grad():
                    try:
                        self.model.to("cpu")
                    except Exception:
                        # Fallback: pull parameters/buffers to CPU
                        for m in self.model.modules():
                            try:
                                for p in m.parameters(recurse=False):
                                    if getattr(p, "is_cuda", False):
                                        p.data = p.data.cpu()
                                for b in m.buffers(recurse=False):
                                    if getattr(b, "is_cuda", False):
                                        b.data = b.data.cpu()
                            except Exception:
                                continue
        except Exception:
            pass

        # Drop references
        try:
            del self.model
        except Exception:
            pass
        self.model = None

        try:
            del self.tokenizer
        except Exception:
            pass
        self.tokenizer = None

        # UI/state cleanup
        try:
            self.chat_messages.clear()
            self._set_diag("t/s: —")
        except Exception:
            pass

        # Restore original CUDA visibility if it was changed
        if getattr(self, "_original_cuda_visible", None) is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._original_cuda_visible
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        # Double GC + per-GPU flush
        try:
            gc.collect()
            self._flush_all_cuda_caches()
            time.sleep(0.05)
            gc.collect()
            self._flush_all_cuda_caches()
        except Exception:
            pass

        self._set_status("Unloaded.")
        try:
            messagebox.showinfo("Unloaded", "Model and tokenizer cleared; VRAM flushed on all visible GPUs.")
        except Exception:
            pass


    def unload_and_restart(self):
        """
        Nuclear option: fully restart the Python process to guarantee CUDA context
        teardown on ALL GPUs (works regardless of sharding/quantization).
        """
        try:
            self._set_status("Unloading & restarting…")
            # Stop any generation thread
            try:
                self.on_stop()
                t = getattr(self, "generation_thread", None)
                if t and t.is_alive():
                    t.join(timeout=3.0)
            except Exception:
                pass

            # Drop big refs so the interpreter has less to serialize/inspect
            self.model = None
            self.tokenizer = None
            try:
                gc.collect()
                self._flush_all_cuda_caches()
            except Exception:
                pass

            # Preserve any CUDA_VISIBLE_DEVICES a user selected earlier
            env = os.environ.copy()
            try:
                messagebox.showinfo(
                    "Restarting",
                    "Restarting the app to fully release VRAM on all GPUs…"
                )
            except Exception:
                pass
            os.execve(sys.executable, [sys.executable] + sys.argv, env)

        except Exception as e:
            try:
                messagebox.showerror("Restart failed", str(e))
            finally:
                self._set_status("Idle.")


    def on_stop(self):
        # Signal the stopper used by generate()
        try:
            self.stop_generation.set()
        except Exception:
            pass

        # Politely tell the streamer to end if it supports it (newer HF versions)
        s = getattr(self, "_current_streamer", None)
        if s is not None and hasattr(s, "end"):
            try:
                s.end()
            except Exception:
                pass

        # Give the worker a brief moment to unwind so GPU kernels exit quickly
        t = getattr(self, "_gen_worker", None)
        if t and t.is_alive():
            try:
                t.join(timeout=0.5)
            except Exception:
                pass



def main():
    app = LLMLabGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
