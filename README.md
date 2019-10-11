# YoloV3-Libtorch1.2-Windows10-VS2017
YoloV3 deploy

OS:Windows10
Compiler:VS2017
LibTorch:1.2


VS2017 need Modify the configuration
VS2017 need Modify the configuration
VS2017 need Modify the configuration

change the "conformance mode/符合模式" property to No. You can find that option in C/C++->Language. That's a new feature in vs2017 and VS2019.

meanwhile VS2015 will error with :
libtorch/include/torch/csrc/utils/variadic.h(195): error C2951: 模板 声明只能在全局、命名空间或类范围内使用


model and config file need download by youself.


reference
https://github.com/walktree/libtorch-yolov3