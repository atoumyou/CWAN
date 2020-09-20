# Color-wise Attention Network for Low-light Image Enhancement
Yousef Atoum, Mao Ye, Liu Ren, Ying Tai, and Xiaoming Liu
 
# Introduction
The CWAN netwrok is as detailed in the CVPRW 2020 paper:
https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Atoum_Color-Wise_Attention_Network_for_Low-Light_Image_Enhancement_CVPRW_2020_paper.pdf

Our framework is implemented and tested with Matlab 2017b, using NVIDIA GTX1080Ti GPU with MatConvNet toolbox.

# Prerequisites
- Tested on Windows only
- Matlab R2017b
- MatConvNet (Latest version)

> matlab/vl_setupnn.m

> matlab/vl_compilenn.m

# Training and Testing:
We provide our models CWAN_L and CWAN_AB which are used jointly to perform color-wise low-light image enhancement. For a quick test of the model, please run the following script:

> Demo_CWAN.m

Note that "run_CWAN_AB_color.m" and "run_CWAN_Lightness.m" can also be used to train the CWAN model within the MatConvNet toolbox (Replace with vl_simplenn).

# Cite
If you utilize this framework, please cite our CVPRW 2020 paper.

@inproceedings{atoum2020color,

  title={Color-wise Attention Network for Low-light Image Enhancement},
  
  author={Atoum, Yousef and Ye, Mao and Ren, Liu and Tai, Ying and Liu, Xiaoming},
  
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  
  pages={506--507},
  
  year={2020}
  
}

# Contact
For questions feel free to post here or directly contact the author at: atoumyou@msu.edu, atoumyou@gmail.com
