# FLINet: Fuzzy Logic for Infrared Small Target Detection
---
Yang Zhang, Meibin Qi, Kunyuan Li, Xiaohong Li, Shuo Zhuang*

### April 22, 2026
Our paper has been accepted by TGRS.

## Overall Framework
<img width="1079" height="638" alt="2aa01c9a-c99f-4df6-97f7-bf71bdf83d73" src="https://github.com/user-attachments/assets/4ed04433-abcf-4aa7-8c5b-c35d4c0204cd" />

## Main Contribution
1) We propose FLINet, a fuzzy-logic-guided deep network that introduces a fuzzy set perspective to IRSTD, providing an uncertainty-aware skip fusion framework for robust target–background separation and improved morphology preservation.
2) We design the GLF computation module that performs dual-branch fuzzy reasoning at local and global levels, enabling graded confidence modeling for cross-level feature selection and offering interpretable membershipbased masks.
3) We introduce a DAF fusion module that exploits interlevel discrepancy to guide cross-scale integration, facilitating adaptive fusion of shallow spatial details and deep semantic cues, and improving performance in challenging cluttered scenes.

### Our project has the following structure:
```
├──./datasets/
  │    ├── IRSTD-1K
  │    │    ├── images
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  |    |    |    ├── ········
  │    │    ├── masks
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  |    |    |    ├── ········
  │    │    ├── train.txt
  │    │    │── test.txt
  │    ├── NUDT-SIRST
  │    │    ├── images
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  |    |    |    ├── ········
  │    │    ├── masks
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  |    |    |    ├── ········
  │    │    ├── train.txt
  │    │    │── test.txt
  │    ├── NUAA-SIRST
  │    │    ├── images
  │    │    │    ├── Misc_1.png
  │    │    │    ├── Misc_2.png
  |    |    |    ├── ········
  │    │    ├── masks
  │    │    │    ├── Misc_1.png
  │    │    │    ├── Misc_2.png
  |    |    |    ├── ········
  │    │    ├── train.txt
  │    │    │── test.txt
```
## Special thanks：
This code is highly borrowed from [SCT](https://github.com/xdFai/SCTransNet).Thanks to Shuai Yuan.
This code is highly borrowed from [FLRSeg](https://github.com/YangLeiSX/FLRSeg).Thanks to Lei Yang.

