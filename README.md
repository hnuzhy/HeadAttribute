# HeadAttribute
A label tool for annotating human head (belonging to rigid objects) pose in 2D image with standard 3D cube

## Introduction 
- this project is based the popular image annotation tool https://github.com/HumanSignal/labelImg
- this is a simple demo tool for answering my issue https://github.com/cvat-ai/cvat/issues/3387 under the `CVAT` project
- it can be used for annotating multiple human head locations and 3D poses from only 2D images
- the current [`labelImg.exe`](https://github.com/hnuzhy/HeadAttribute/releases/tag/v1.0) file can only be executed on Windows, not supporting Linux/Ubuntu/Mac
- please download the released file [`labelImg.exe`](https://github.com/hnuzhy/HeadAttribute/releases/tag/v1.0) and put it under the subfolder [`./dist/`](./dist/) for running
- possible local workdir is `C:\Users\zhy19\AppData\Local\Temp\_MEI1731402` when running `labelImg.exe`
- you can change the static 3D head model [`./dist/data/Female3DHead.obj`](./dist/data/Female3DHead.obj) into your ideal one

## Running Advice
If you just want to use this annotation tool for labeling human head poses, thes just run `labelImg.exe` and follow user manuals of original [labelImg](https://github.com/HumanSignal/labelImg). Below is an example video of using the tool.

https://github.com/user-attachments/assets/4b413102-c61e-46f6-b027-ce595c14acee


## Compiling Advice

## References
Citation of labelImg: `Tzutalin. LabelImg. Git code (2015). https://github.com/tzutalin/labelImg`

Citation of HeadAttribute: This project is also based on my previous work for [`simultaneous head detection and pose estimation`](https://arxiv.org/abs/2212.03623)
```
@article{zhou2022intuitive,
  title={An Intuitive and unconstrained 2D cube representation for simultaneous head detection and pose estimation},
  author={Zhou, Huayi and Jiang, Fei and Xiong, Lili and Lu, Hongtao},
  journal={arXiv preprint arXiv:2212.03623},
  year={2022}
}
```

