### 2018-11-26 重要问题修复
发现了resnet18里的一个重要问题，见下面两张图
<center class="half">
    <img src="https://github.com/TerryBryant/MyWork/blob/master/caffe/images/resnet18_before.png">
    <img src="https://github.com/TerryBryant/MyWork/blob/master/caffe/images/resnet18_after.png">
</center>

```res2a_branch2b_relu```是没有的！！！之前参考的网上的netscope图编写的python代码，我没有注意到这一点，因为白色的实在是很容易忽略。
现在才知道为什么大神们会把conv、bn、scale和relu分开画了，因为netscope在显示第四行的时候会变成白色
