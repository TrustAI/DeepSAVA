import tensorflow as tf
import numpy as np
from scikit_opt.sko.GA import GA
from scikit_opt.sko.SA import SA_TSP
from bayes_opt import BayesianOptimization
import random
def initRandom(constraint):
    #产生随机初始值，由列表表示权值的个数，每个权值的上下界由一个元组表示
    #ex.constraint = [(1,20),(13, 21)], 返回np.array
    weight = np.array([])
    for i in constraint:
        weight = np.append(weight, (i[1] - i[0]) * np.random.random() + i[0])
    return weight

def changeWeight( now ):
    '''产生扰动后的权值，返回np.array
    :param constraint: 权值约束 :param changeRange: 权值扰动的范围
    :param bias: 权值扰动的偏好方向,eg: 0.1代表更倾向偏大的值20%
    :param now: 现有权值
    '''
    changeRange=1
    bias=0
    constraint=[-np.pi/3,np.pi/3]
    result = np.copy(now)
    for index in range(len(result)):
        delta = (np.random.random() - 0.5 + bias) * changeRange
        while (delta + result[index] > constraint[index][1] or
            delta + result[index] < constraint[index][0]):
            delta = (np.random.random() - 0.5 + bias) * changeRange
        result[index] += delta
    return result

def randf_in(now):
        import time
        new = np.copy(now)
        if now ==1:
            new =0
        else:
            new =1
        
        return new


class SAoptimizer:
    def __init__(self):
        super().__init__()

    def optimize(self,lb,ub,f, initf, randf_float,randf_int ,type,n_dim = 0,
            t=10000, alpha=0.98, stop=1e-1, iterPerT=1, l=1):
        '''
        :param f: 目标函数,接受np.array作为参数 :param ybound: y取值范围
        :param initf: 目标函数的初始权值函数，返回np.array :param alpha: 退火速率
        :param iterPerT: 每个温度下迭代次数 :param t: 初始温度 :param l:新旧值相减后的乘数，越大，越不容易接受更差值
        :param stop: 停火温度 :param randf: 对参数的随机扰动函数，接受现有权值，返回扰动后的新权值np.array
        '''
        #初始化
        y_old = None
        while y_old == None or y_old < lb or y_old > ub:
            x_old = initf
            y_old = f(x_old)
        y_best = y_old
        x_best = np.copy(x_old)
        #降温过程
        x_new = np.copy(x_old)
        count = 0
        while(t > stop):
            downT = False
            for i in range(iterPerT):
                for i in range(len(initf)):
                    if type[i] == 'float':
                        x_new[i] = randf_float(x_old[i])
                    else:
                        x_new[i] = randf_int(x_old[i])
               
                y_new = f(x_new)
                if y_new > ub or y_new < lb:
                    continue
                #根据取最大还是最小决定dE,最大为旧值尽可能小于新值
                dE = -(y_old - y_new) * l
                if dE < 0:
                    downT = True
                    count = 0
                else: count += 1
                if self.__judge__(dE, t):
                    x_old = x_new
                    y_old = y_new
                    if y_old < y_best:
                        y_best = y_old
                        x_best = x_old
                                   
            if downT:
                t = t * alpha
                #长时间不降温
            if count > 1000: break
        self.weight = x_best
        return y_best
    def __judge__(self, dE, t):
        '''
        :param dE: 变化值\n
        :t: 温度\n
        根据退火概率: exp(-(E1-E2)/T)，决定是否接受新状态
        '''
        if dE < 0:
            return 1
        else:
            p = np.exp(-dE / t)
            
                    
            if p > np.random.random(size=1):
                return 1
            else: return 0

        
                    

 
        
def my_sa(
    loss, seq_len, indicator,theta,feed_dict=None,
    sa_extra_kwargs=None, sess=None
):

    def tf_run(x):
        """Function to minimize as provided to ``scipy.optimize.fmin_l_bfgs_b``.
        Args:
            x (np.ndarray): current flows proposal at a given stage of the
                            optimization (flattened `np.ndarray` of type
                            `np.float64` as required by the backend FORTRAN
                            implementation of L-BFGS-B).
        Returns:
            `Tuple` `(loss, loss_gradient)` of type `np.float64` as required
            by the backend FORTRAN implementation of L-BFGS-B.
        """
        
        
        x = np.reshape(x, (seq_len*2))
        feed_dict.update({indicator: x[0:seq_len],theta:x[seq_len:seq_len*2]})
        loss_val = sess_.run(
            [loss], feed_dict=feed_dict
        )
        loss_val = np.sum(loss_val).astype(np.float64)
        

        return loss_val

    

    if feed_dict is None:
        feed_dict = {}
    if sa_extra_kwargs is None:
        sa_extra_kwargs = {}
    
    init = []
    init.append(np.random.randint(2, size=seq_len))
    init.append(np.random.randn(seq_len))
    print(np.reshape(np.array(init),(seq_len*2)))
    sa_kwargs = {
        'f': tf_run,
        'randf_int':randf_in,
        'initf': np.reshape(np.array(init),(seq_len*2)),
        'randf_float': changeWeight,
        'n_dim':seq_len*2,
        'lb':-np.inf,
        'ub':np.inf,
        'type' :np.repeat(['int','float'], seq_len),
        'stop':1e-5,
        't':1e3,
        'alpha':0.99,
        'l':10,
        'iterPerT':1
        

       
    }

    

    # define the default extra arguments to fmin_l_bfgs_b


    sa_kwargs.update(sa_extra_kwargs)
   

    sess_ = tf.Session() if sess is None else sess
    
    
    sa = SAoptimizer()
    ans = sa.optimize(**sa_kwargs)
    if sess is None:
        sess_.close()
    return sa.weight, ans
def gen_a(
    loss, seq_len, indicator,theta,feed_dict=None,
    ga_extra_kwargs=None, sess=None
):
    
    def tf_run(x):
        """Function to minimize as provided to ``scipy.optimize.fmin_l_bfgs_b``.
        Args:
            x (np.ndarray): current flows proposal at a given stage of the
                            optimization (flattened `np.ndarray` of type
                            `np.float64` as required by the backend FORTRAN
                            implementation of L-BFGS-B).
        Returns:
            `Tuple` `(loss, loss_gradient)` of type `np.float64` as required
            by the backend FORTRAN implementation of L-BFGS-B.
        """
        x = np.reshape(x, (seq_len*2))
        feed_dict.update({indicator: x[0:seq_len],theta:x[seq_len:seq_len*2]})
        loss_val = sess_.run(
            [loss], feed_dict=feed_dict
        )
        loss_val = np.sum(loss_val).astype(np.float64)
        

        return loss_val

    

    if feed_dict is None:
        feed_dict = {}
    if ga_extra_kwargs is None:
        ga_extra_kwargs = {}

    ga_kwargs = {
        'func': tf_run,
        'n_dim':seq_len*2,
        'lb':np.repeat([0,-np.pi/3], seq_len),
        'ub':np.repeat([1,np.pi/3], seq_len),
        'precision': np.repeat([1,1e-7], seq_len)
        

       
    }

    

    # define the default extra arguments to fmin_l_bfgs_b


    ga_kwargs.update(ga_extra_kwargs)
   

    sess_ = tf.Session() if sess is None else sess
    raw_results = GA(**ga_kwargs).run()
    if sess is None:
        sess_.close()

    return {
        'mask': raw_results[0][0:seq_len],
        'angle': raw_results[0][seq_len:seq_len*2],
        'loss': raw_results[1],
    }
    
def gen_am(
    loss, seq_len, indicator,feed_dict=None,
    ga_extra_kwargs=None, sess=None
):
    
    def tf_run(x):
        """Function to minimize as provided to ``scipy.optimize.fmin_l_bfgs_b``.
        Args:
            x (np.ndarray): current flows proposal at a given stage of the
                            optimization (flattened `np.ndarray` of type
                            `np.float64` as required by the backend FORTRAN
                            implementation of L-BFGS-B).
        Returns:
            `Tuple` `(loss, loss_gradient)` of type `np.float64` as required
            by the backend FORTRAN implementation of L-BFGS-B.
        """
        x = np.reshape(x, (seq_len))
        feed_dict.update({indicator: x})
        loss_val = sess_.run(
            [loss], feed_dict=feed_dict
        )
        loss_val = np.sum(loss_val).astype(np.float64)
        

        return loss_val

    

    if feed_dict is None:
        feed_dict = {}
    if ga_extra_kwargs is None:
        ga_extra_kwargs = {}

    ga_kwargs = {
        'func': tf_run,
        'n_dim':seq_len,
        'lb':np.repeat([0], seq_len),
        'ub':np.repeat([1], seq_len),
        'precision': np.repeat([1], seq_len)
        

       
    }

    

    # define the default extra arguments to fmin_l_bfgs_b


    ga_kwargs.update(ga_extra_kwargs)
   

    sess_ = tf.Session() if sess is None else sess
    raw_results = GA(**ga_kwargs).run()
    if sess is None:
        sess_.close()

    return {
        'mask': raw_results[0][0:seq_len],
        'angle': raw_results[0][seq_len:seq_len*2],
        'loss': raw_results[1],
    }

def sa_tsp(
    loss, seq_len, indicator,feed_dict=None,
    satsp_extra_kwargs=None, sess=None
):
    
    def tf_run(x):
        """Function to minimize as provided to ``scipy.optimize.fmin_l_bfgs_b``.
        Args:
            x (np.ndarray): current flows proposal at a given stage of the
                            optimization (flattened `np.ndarray` of type
                            `np.float64` as required by the backend FORTRAN
                            implementation of L-BFGS-B).
        Returns:
            `Tuple` `(loss, loss_gradient)` of type `np.float64` as required
            by the backend FORTRAN implementation of L-BFGS-B.
        """
        x = np.reshape(x, (seq_len))
        feed_dict.update({indicator: x})
        loss_val = sess_.run(
            [loss], feed_dict=feed_dict
        )
        loss_val = np.sum(loss_val).astype(np.float64)
        

        return loss_val

    

    if feed_dict is None:
        feed_dict = {}
    if satsp_extra_kwargs is None:
        satsp_extra_kwargs = {}

    satsp_kwargs = {
        'func': tf_run,
        'x0':np.random.randint(2,size=seq_len),
        'T_max':50,
        'T_min':1e-7,
        'L': 10*seq_len
        

       
    }

    

    # define the default extra arguments to fmin_l_bfgs_b


    satsp_kwargs.update(satsp_extra_kwargs)
   

    sess_ = tf.Session() if sess is None else sess
    best_points, best_distance = SA_TSP(**satsp_kwargs).run()
    if sess is None:
        sess_.close()

    return {
        'mask': best_points,
        'loss': best_distance
    }
def ba_op(
     train,init_varibale_list,true_label_prob,seq_len, indicator,f,feed_dict=None,
    bo_extra_kwargs=None, sess=None):
    
    def tf_run(x):
        """Function to minimize as provided to ``scipy.optimize.fmin_l_bfgs_b``.
        Args:
            x (np.ndarray): current flows proposal at a given stage of the
                            optimization (flattened `np.ndarray` of type
                            `np.float64` as required by the backend FORTRAN
                            implementation of L-BFGS-B).
        Returns:
            `Tuple` `(loss, loss_gradient)` of type `np.float64` as required
            by the backend FORTRAN implementation of L-BFGS-B.
        """
        x = int(np.around(x))
        ind = np.zeros((seq_len))
        ind[x] = 1
        feed_dict.update({indicator: ind})
        sess_.run(tf.initialize_variables(init_varibale_list))
        sess_.run(train, feed_dict=feed_dict)
        prob = sess_.run(
            [true_label_prob], feed_dict=feed_dict
        )
        prob = np.sum(prob).astype(np.float64)
        

        return 1-prob

    
    sess_ = tf.Session() if sess is None else sess
    if feed_dict is None:
        feed_dict = {}
    if bo_extra_kwargs is None:
        bo_extra_kwargs = {}
    input = []
    for i in range(seq_len):
        ind = np.zeros(seq_len)
        ind[i] = 1
        input.append(ind)
    input_ind = np.array(ind)
    bo_kwargs = {
        'f': tf_run,
        'pbounds':{'x': (0,seq_len-1)},
        'verbose':2,
        'random_state':3,

       
    }
    optimizer = BayesianOptimization(**bo_kwargs)

    baop = optimizer.maximize(init_points=10, n_iter=10)



   
    
    if sess is None:
        sess_.close()

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res),file =f)
    index = int(np.around(optimizer.max['params']['x']))
    
    return index
def ba_op_4(
  train,init_varibale_list,true_label_prob,seq_len, indicator,f,feed_dict=None,
 bo_extra_kwargs=None, sess=None):

 def tf_run(x1,x2,x3,x4):
     """Function to minimize as provided to ``scipy.optimize.fmin_l_bfgs_b``.
     Args:
         x (np.ndarray): current flows proposal at a given stage of the
                         optimization (flattened `np.ndarray` of type
                         `np.float64` as required by the backend FORTRAN
                         implementation of L-BFGS-B).
     Returns:
         `Tuple` `(loss, loss_gradient)` of type `np.float64` as required
         by the backend FORTRAN implementation of L-BFGS-B.
     """
     
     ind = np.zeros((seq_len))
     for x in [x1,x2,x3,x4]:
        i = int(np.around(x))
        ind[i] = 1
     feed_dict.update({indicator: ind})
     sess_.run(tf.initialize_variables(init_varibale_list))
     sess_.run(train, feed_dict=feed_dict)
     prob = sess_.run(
         [true_label_prob], feed_dict=feed_dict
     )
     prob = np.sum(prob).astype(np.float64)


     return 1-prob


 sess_ = tf.Session() if sess is None else sess
 if feed_dict is None:
     feed_dict = {}
 if bo_extra_kwargs is None:
     bo_extra_kwargs = {}
 input = []
 for i in range(seq_len):
     ind = np.zeros(seq_len)
     ind[i] = 1
     input.append(ind)
 input_ind = np.array(ind)
 bo_kwargs = {
     'f': tf_run,
     'pbounds':{'x1': (0,seq_len-4),'x2': (1,seq_len-3),'x3':(2,seq_len-2),'x4':(3,seq_len-1)},
     'verbose':1,
     'random_state':3,

    
 }
 optimizer = BayesianOptimization(**bo_kwargs)

 baop = optimizer.maximize(init_points=10, n_iter=10,)




 
 if sess is None:
     sess_.close()
 for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res),file =f)
 ind1,ind2,ind3,ind4 = int(np.around(optimizer.max['params']['x1'])),int(np.around(optimizer.max['params']['x2'])),int(np.around(optimizer.max['params']['x3'])),int(np.around(optimizer.max['params']['x4']))

 return ind1,ind2,ind3,ind4
