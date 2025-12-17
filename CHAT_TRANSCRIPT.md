# picunic Chat Transcript

> In this empty rust project, I plan to create an image-to-unicode converter lib and exe. Unlike simpler statistical methods elsewhere, it will use a CNN to find best mapping from chunks of image to 1 unicode each. wdyt? what arch is suitable



> the embedding appraoch sounds better for reusability. But which approach is faster for inference?



> is 80x24 the traditional size of terminal chars?



> Let's start with a proof of concept. And remember that for this to work, you need fixed width font, some modern mono fonts are not strictly fixed width

At this point, it implemented a PoC that maps each image chunk to the unicode directly using a closeness function

> Maybe using python for training will actually be easier? then after training we can save the model and load it in rust and use it for inference



> we are at 1k python and 1k rust code. Do a quick check to see if you can cleanup a few things and try to not add dependecies for trivial stuff. to make it minimalist, concise and easily readable and understandble. Then proceed with training

reduced from 2000 to 700 lines.

> what if I want to optimize for edge detection correctly

added another channel to the input that has the image pre-processed for edge detection

> why is it so slow. test_image.png with default settings takes ~6.5s

Cursor correctly pointed out I was using debug build

> The perforamnce is not satisfactory. In dataset augment, can you add oneline comment on what each transformation is doing. I don't think the additional channel for edge detection is necessary. What you are calling luminance should actually learning shapes/edge/angles etc. either we're learning too slow, too little data for the model size, embedding space not enough or too big, or what? any insights? the accuracy is <50%



> can you quickly output images of one batch to a file so I can inspect



> The render function writes a character about in the middle of a square. If we want to draw with characters, we want them to fill their box. Terminal characters tend to be 1:2 aspect ratio. make the training images of this aspect ratio and avoid padding around (not including padding within a character, e.g. letter n is shorter than N so it will have some empty space above it), curate a list of non-similar unicode chars, and go back to embeddings

It incorrectly stretched each character to fill the rectangle


> I didn't mean to stretch the chars. I meant do not draw them leaving margins arond to center the char, instead draw it so that it fills the same space of the 16x8 window as it would on the terminal character. So that when we map image to unicode and print on terminal, each char will naturally align with the chunk it represents



> go on

At this point, a connection error happened. Instead of the last prompt, it went do something else. In the next prompt, the pipes are an artifact of copying from cursor-cli



> wait. When I said go on, I meant these instructions: Use the learned embeddings to learn which unicode chars are distinct. Basically, get a very large charset of unicode and map to               │
 │ embeddings, then the (normalized) dot product of embeddings will tell you how similar they are. Then select a subset that is below a          │
 │ threshold of pairwise distinctness, then retrain the model with this subset (which would be larger than our original distinct set, and        │
 │ give better results). Did you do this? the onnx feature was working fine without all that dylib path stuff, what changed. Also, let's simplify things. Let's make it do one thing well, no need for that NCC mode. Only mode should be using CNN with embeddings.



> does this exporting needed because you removed onnx from cargo dependencies?



> In the rust, side keep the option to only ascii during inference

> cleanup code, documentation and assets for commit                                                                                                                                                            │


