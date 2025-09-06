Don't you hate it when a brand new model drops, but all the actually-good backends take a while to add support?  Well now you don't have to wait 
for the LCPP people to get off their butts and add new arches.  This janky vibeslop will load anything that is supported by transformers and allow 
you to pester it with a friendly pointy-clicky graphical interface instead of manually fiddling with python scripts like some kind of cave person. 

Usage should be fairly self-explanatory.

![gui1](https://github.com/user-attachments/assets/734b83f4-a3ca-4f24-b56d-cc4616f243a8)

![gui2](https://github.com/user-attachments/assets/260b49f1-0c1c-4db2-8a89-64675cd8898e)


Dependencies:  transformers and torch are the only hard requirements.  You will probably want bitsandbytes, flash_attn, and huggingface_hub as 
well for the various functions that won't work if you don't have them.  You probably already have tkinter, but if you don't, you'll need that too.

Known issues: It's ass-achingly slow.  I don't know if FA2 is actually working or not, but probably not since the speed is abysmal even with it 
enabled.  The unload-model function rarely works properly with a multi-GPU setup, so you will need to restart the program to fully clear out 
the VRAM across the board if you have multiple cards.  The "reasoning mode" thinking filter doesn't seem to work.  Maybe it works with SOMETHING, 
but not with any of the thinking models I've tried.  It generally doesn't respect the GPU selection check boxes unless you use the "reset with 
selected GPUs" button.  Cuda doesn't work for me (could be a me-problem) in native windows.  Works fine in WSL2 (and presumably real linux) though.
