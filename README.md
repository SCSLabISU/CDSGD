# CDSGD
This is a placeholder repository for Consensus Based Distributed Stochastic Gradient Descent. For more details, please see the paper: 
**[Collaborative Deep Learning in Fixed Topology Networks][1]**  
Zhanhong Jiang, Aditya Balu, Chinmay Hegde, Soumik Sarkar

### Usage
python main.py -m CNN -b 512 -ep 200 -d cifar10 -n 5 -cp 1 -g 3 CDSGD

-m is the model name which is CNN, FCN and Big_CNN; -b batchsize; -ep is epochs; -d is the dataset; -n is the no. of agents; -cp is communication period; -g is the GPU is you want to use. Say you have 4 gpus, then you choose which gpu to use. Then finally the experiments you wanna run. SGD, CDSGD, EASGD, CDMSGD, MSGD, FASGD etc.

### License

BSD

[1]:https://arxiv.org/abs/1706.07880
[2]:http://scslab.me.iastate.edu/index.html

