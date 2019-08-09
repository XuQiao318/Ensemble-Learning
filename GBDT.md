# GBDT算法梳理

	![](https://i.imgur.com/Y2Xe9Bs.png)

- 梯度下降树
	- gbdt 是通过采用加法模型（即基函数的线性组合），以及不断减小训练过程产生的残差来达到将数据分类或者回归的算法。
- GBDT的训练过程
	![](https://i.imgur.com/xuIGkC3.png)

- gbdt通过多轮迭代,每轮迭代产生一个弱分类器，每个分类器在上一轮分类器的残差基础上进行训练。对弱分类器的要求一般是足够简单，并且是低方差和高偏差的。因为训练的过程是通过降低偏差来不断提高最终分类器的精度，（此处是可以证明的）。

- 弱分类器一般会选择为CART TREE（也就是分类回归树）。由于上述高偏差和简单的要求 每个分类回归树的深度不会很深。最终的总分类器 是将每轮训练得到的弱分类器加权求和得到的（也就是加法模型）。
- 
- GBDT是一个应用很广泛的算法，可以用来做分类、回归。在很多的数据上都有不错的效果。GBDT这个算法还有一些其他的名字，比如说MART(Multiple Additive Regression Tree)，GBRT(Gradient Boost Regression Tree)，Tree Net等，其实它们都是一个东西（参考自wikipedia – Gradient Boosting)，发明者是Friedman

- Gradient Boost其实是一个框架，里面可以套入很多不同的算法，可以参考一下机器学习与数学(3)中的讲解。Boost是"提升"的意思，一般Boosting算法都是一个迭代的过程，每一次新的训练都是为了改进上一次的结果。

- 原始的Boost算法是在算法开始的时候，为每一个样本赋上一个权重值，初始的时候，大家都是一样重要的。在每一步训练中得到的模型，会使得数据点的估计有对有错，我们就在每一步结束后，增加分错的点的权重，减少分对的点的权重，这样使得某些点如果老是被分错，那么就会被“严重关注”，也就被赋上一个很高的权重。然后等进行了N次迭代（由用户指定），将会得到N个简单的分类器（basic learner），然后我们将它们组合起来（比如说可以对它们进行加权、或者让它们进行投票等），得到一个最终的模型。


## 一、前向分布算法

- 加法模型

	![](https://i.imgur.com/3xnLveg.png)
- 通常这是一个复杂的优化问题。前向分步算法求解这一优化问题的想法是：因为学习的是加法模型，如果能够从前向后，每一步只学习一个基函数及其系数，逐步逼近优化目标函数，那么就可以简化优化的复杂度，具体的，每步只需优化如下损失函数。
	![](https://i.imgur.com/trLpldI.png)

- 流程

	![](https://i.imgur.com/B5GJ2E3.png)

	![](https://i.imgur.com/yie5LNx.png)

- 值得一提的是，Adaboost算法是前向分步算法的特例，当其损失函数是指数损失函数时，就是Adaboost算法。


## 二、负梯度拟合

- Gradient Boost与传统的Boost的区别是，每一次的计算是为了减少上一次的残差(residual)，而为了消除残差，我们可以在残差减少的梯度(Gradient)方向上建立一个新的模型。所以说，在Gradient Boost中，每个新的模型的简历是为了使得之前模型的残差往梯度方向减少，与传统Boost对正确、错误的样本进行加权有着很大的区别。

- GBDT拟合残差，为什么是负梯度拟合？

![](https://i.imgur.com/NxSWx6G.png)


![](https://i.imgur.com/h1sEd3u.png)


![](https://i.imgur.com/92Q4eBn.png)


![](https://i.imgur.com/dbK76oT.png)


## 三、损失函数

![](https://i.imgur.com/nadtb0A.jpg)

![](https://i.imgur.com/FDVOFil.png)

## 四、回归

- GBDT之回归树

![](https://i.imgur.com/2MJcMSH.png)

![](https://i.imgur.com/rvAirny.png)

![](https://i.imgur.com/2NWtoMf.png)



## 五、二分类，多分类

- GBDT之分类树

 	+ 在分类问题中，有一个很重要的内容叫做Multi-Class Logistic，也就是多分类的Logistic问题，它适用于那些类别数>2的问题，并且在分类结果中，样本x不是一定只属于某一个类可以得到样本x分别属于多个类的概率（也可以说样本x的估计y符合某一个几何分布），这实际上是属于Generalized Linear Model中讨论的内容，这里就用一个结论：如果一个分类问题符合几何分布，那么就可以用Logistic变换来进行之后的运算。
 	+ GBDT分类算法思想上和GBDT的回归算法没有什么区别，但是由于样本输出不是连续值，而是离散类别，导致我们无法直接从输出类别去拟合类别输出误差。为了解决这个问题，主要有两种方法。一是用指数损失函数，此时GBDT算法退化为AdaBoost算法。另一种方法是用类似于逻辑回归的对数似然损失函数的方法。也就是说，我们用的是类别的预测概率值和真实概率值的差来拟合损失。当损失函数为指数函数时，类似于AdaBoost算法，这里不做介绍，下面介绍损失函数为log函数时的GBDT二分类和多分类算法。

![](https://i.imgur.com/XWCIqMu.png)

![](https://i.imgur.com/5iVnk0u.png)

![](https://i.imgur.com/Hlc9BLF.png)


## 六、正则化
- 1 CART树的剪枝


- 2 设置步长η, 。参数：Shrinkage–>(0, 1]
即在每一轮迭代获取最终学习器的时候按照一定的步长进行更新。其中0<η≤1是步长，对于同样的训练集学习效果，较小的η意味着需要更多的迭代次数；通常用步长和迭代最大次数一起来决定算法的拟合效果。参数：Shrinkage
注：步长如何设置才能防止过拟合？
因为较小的η意味着需要更多的迭代次数，防止过拟合就要稍微加大步长，用步长和迭代最大次数一起来决定算法的拟合效果。

- 3 子采样。
参数subsample–>(0, 1]。这里是不放回随机抽样。选择小于1的比例可以减少方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低，一般在[0.5, 0.8]之间。	GBDT子采样的过程：比如有100个样本，subsample=0.8，第一棵树训练时随机从100个样本中抽取80%，有80个样本训练模型；第二棵树再在100个样本再随机采样80%数据，也就是80个样本，训练模型；以此类推。

- GBDT降低学习率可以实现正则化效果呢？

	+ 因为一般根据在神经网络的经验而言，降低学习率，可以实现更高的训练效果，即进一步拟合；

	+ 在gbdt中，这个学习率与神经网络中的学习率担任的角色不一样；

	+ gbdt中的学习率主要是调节每棵树的对预测结果的贡献；如果学习率下降，就降低了每棵树的贡献；模型训练的预测效果就会下降；为了达到和高学习率相同的效果，就需要生成更多的树；

	+ 当时的疑惑是如果下降学习率，那么就会生成更多的树，就会更加拟合；怎么会有正则化效果呢？

	+ 因为下降学习率，并没有增加更多的树，前提假设其他的超参是不变的；

	+ 在学习率等超参数固定的情况下，树的数量越多，就模型训练精度越高； 

	+ 在树的数量等超参数固定的情况下， 学习率越高，模型训练精度越高；


## 七、优缺点

- 优点：

	- 预测精度高
	- 适合低维数据
	- 能处理非线性数据
	- 可以灵活处理各种类型的数据，包括连续值和离散值。
	- 在相对少的调参时间情况下，预测的准备率也可以比较高。这个是相对SVM来说的。
	- 使用一些健壮的损失函数，对异常值的鲁棒性非常强。比如 Huber损失函数和Quantile损失函数。
- 缺点：
	- 由于弱学习器之间存在依赖关系，难以并行训练数据。不过可以通过自采样的SGBT来达到部分并行。
	- 如果数据维度较高时会加大算法的计算复杂度

## 八、sklearn参数

- 分类

>>> from sklearn.datasets import make_hastie_10_2
>>> from sklearn.ensemble import GradientBoostingClassifier

>>> X, y = make_hastie_10_2(random_state=0)
>>> X_train, X_test = X[:2000], X[2000:]
>>> y_train, y_test = y[:2000], y[2000:]

>>> clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
...     max_depth=1, random_state=0).fit(X_train, y_train)
>>> clf.score(X_test, y_test)                 
0.913...

- 回归

>>> import numpy as np
>>> from sklearn.metrics import mean_squared_error
>>> from sklearn.datasets import make_friedman1
>>> from sklearn.ensemble import GradientBoostingRegressor

>>> X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)
>>> X_train, X_test = X[:200], X[200:]
>>> y_train, y_test = y[:200], y[200:]
>>> est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
... max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
>>> mean_squared_error(y_test, est.predict(X_test))
5.00...



- 参数


- 因基分类器是决策树，所以很多参数都是用来控制决策树生成的，这些参数与前面决策树参数基本一致，对于一致的就不进行赘述。


- loss:损失函数度量，有对数似然损失deviance和指数损失函数exponential两种，默认是deviance，即对数似然损失，如果使用指数损失函数，则相当于Adaboost模型。


- criterion: 样本集的切分策略，决策树中也有这个参数，但是两个参数值不一样，这里的参数值主要有friedman_mse、mse和mae3个，分别对应friedman最小平方误差、最小平方误差和平均绝对值误差，friedman最小平方误差是最小平方误差的近似。


- subsample:采样比例，这里的采样和bagging的采样不是一个概念，这里的采样是指选取多少比例的数据集利用决策树基模型去boosting，默认是1.0，即在全量数据集上利用决策树去boosting。


- warm_start:“暖启动”，默认值是False，即关闭状态，如果打开则表示，使用先前调试好的模型，在该模型的基础上继续boosting，如果关闭，则表示在样本集上从新训练一个新的基模型，且在该模型的基础上进行boosting。


- 属性/对象

- feature_importance_:特征重要性。


- oob_improvement_:每一次迭代对应的loss提升量。oob_improvement_[0]表示第一次提升对应的loss提升量。


- train_score_:表示在样本集上每次迭代以后的对应的损失函数值。


- loss_:损失函数。


- estimators_：基分类器个数。


- 方法


- apply(X)：将训练好的模型应用在数据集X上，并返回数据集X对应的叶指数。


- decision_function(X):返回决策函数值（比如svm中的决策距离）



- fit(X,Y):在数据集（X,Y）上训练模型。



- get_parms():获取模型参数



- predict(X):预测数据集X的结果。



- predict_log_proba(X):预测数据集X的对数概率。



- predict_proba(X):预测数据集X的概率值。



- score(X,Y):输出数据集（X,Y）在模型上的准确率。



- staged_decision_function(X):返回每个基分类器的决策函数值


- staged_predict(X):返回每个基分类器的预测数据集X的结果。



- staged_predict_proba(X):返回每个基分类器的预测数据集X的概率结果。



## 九、应用场景

- 该版本GBDT几乎可用于所有回归问题（线性/非线性），相对logistic regression仅能用于线性回归，GBDT的适用面非常广。亦可用于二分类问题（设定阈值，大于阈值为正例，反之为负例）。

## 参考链接

1、https://www.cnblogs.com/bnuvincent/p/9693190.html

2、https://www.zhihu.com/question/63560633/answer/581670747

3、https://blog.csdn.net/w5688414/article/details/78002064

4、https://blog.csdn.net/qq_24519677/article/details/82020863

5、https://blog.csdn.net/weixin_40363423/article/details/98878382

6、https://blog.csdn.net/ningyanggege/article/details/87974691

7、https://blog.csdn.net/suv1234/article/details/72588048