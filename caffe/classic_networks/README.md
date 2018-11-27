## 2018-11-26 重要问题修复
发现了resnet18里的一个重要问题，见下面两张图
<center class="half">
    <img src="https://github.com/TerryBryant/MyWork/blob/master/caffe/images/resnet18_before.png">
    <img src="https://github.com/TerryBryant/MyWork/blob/master/caffe/images/resnet18_after.png">
</center>

论文里```res2a_branch2b_relu```是没有的！！！之前参考的网上的netscope图编写的python代码，我没有注意到这一点，因为白色的实在是很容易忽略。
现在才知道为什么大神们会把conv、bn、scale和relu分开画了，因为netscope在显示第四行的时候会变成白色

## 2018-11-27 加入se-resnet18
参考的是[这里](https://github.com/shicai/SENet-Caffe)的实现，其中有个地方是需要注意的，就是```res2a_/scale1```这里，注意两个bottom的
顺序，这一层主要是把一个1x64的数据乘到1x64x56x56的feature map上，得到1x64x56x56的输出，如果顺序反了，输出就变成1x64了。具体原因要参考caffe源码了
