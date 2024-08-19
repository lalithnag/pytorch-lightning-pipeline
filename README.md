### Pytorch lightning utilities

In this repo, I am providing utilities and helper function for a basic deep learning project, for whomever needs it. 
Some basic UNet model implementations are written with lightning, these can be used in normal Pytorch as well as extended for other models.
Other data processing and training code is also provided. It contains code for things like:

* Splitting data, performing stratified sampling, and checking for data leaks
* Dataloader for monoscopic and stereoscopic data with and without masks, with pre-processing
* The dataloaders are written as PL datamodules
* Pl model implementations for a basic UNet segmentation model and a depth estimation model based on [Monodepth2](https://github.com/nianticlabs/monodepth2). Some of the code is also built on top of the Monodepth2 repo, so refer to their work to understand it in more detail.
* Please note that the lightning models can also be used in native pytorch without change in workflow, which is cool.
* Lightning modules for training.
* Some other I/O utils (including sending yourself a telegram message after model training completes lol).
