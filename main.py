import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import timeit
from tqdm import tqdm
class number_recognition():
    def __init__(self,iteration,dataname,n_p,n_l1,n_o):
        self.iteration=iteration
        self.dataname=dataname
        self.num_of_par=n_p*n_l1+n_l1+n_l1*n_o+n_o
        self.n_o=n_o
        self.n_l1=n_l1
        self.n_p=n_p
        self.n_in_l1=self.n_p*self.n_l1
        self.n_in_l_b=self.n_p*self.n_l1+self.n_l1
        self.n_till_o=self.n_p*self.n_l1+self.n_l1+self.n_l1*self.n_o
        #print n_p,n_l1,n_o,self.n_in_l1,self.n_in_l_b,self.n_till_o,self.num_of_par-self.n_till_o
        self.dropout_rate=0
    def drop_out(self,drop_rate):
        self.dropout_rate=drop_rate        
    def get_data(self):
        self.data_raw=pd.read_csv(dataname).values
    
    def get_train_test(self):


        train_label=self.data_raw[:5000,0]
        self.train_pixel=self.data_raw[:5000,1:]/255.00
        self.test_pixel=self.data_raw[5000:6000,1:]/255.00
        self.test_label=self.data_raw[5000:6000,0]

        
        self.d_true=np.zeros(( len(train_label),10))
        self.con_mat=(self.train_pixel[:,:]).reshape(5000,1,28,28)
        
        self.con_test=(self.test_pixel[:,:]).reshape(1000,1,28,28)

        for bb in range(len(train_label)):
            self.d_true[bb,int(train_label[bb] )]=1
        
    
    def convolution(self,num_feature,lx_f):
        if lx_f==5:
            kernelst=[self.feature_5_gussian_blur(),\
			    self.feature_5_unsharp_masking()\
			    , self.feature_5_ones() ,\
			    self.feature_5_gradient1(),\
			    self.feature_5_gradient2(),\
			    self.feature_5_sobel1(),\
			    self.feature_5_sobel2()]
            kernels=kernelst[:num_feature]

        elif lx_f==3:
            kernelst=[self.feature_3_identity(), self.feature_3_edge() , self.feature_3_sharpen(), self.feature_3_guassian_blur()\
                     , self.feature_3_box_blur() ,\
		     self.feature_3_top_sobel(),\
		     self.feature_3_left_sobel(),\
		     self.feature_3_right_sobel(),\
		     self.feature_3_bottom_sobel(),\
		     self.feature_3_outline(),\
		     self.feature_3_emboss(),\
		     self.feature_3_gradient1(),\
		     self.feature_3_gradient2(),\
		     self.feature_3_edge1(),\
		     self.feature_3_edge2() ]
            kernels=kernelst[:num_feature]

        lp=(np.shape(self.con_mat))[1]
        lx=(np.shape(self.con_mat))[2]
        ly=(np.shape(self.con_mat))[3]
        res=np.zeros((5000,num_feature*lp,lx-lx_f+1,ly-lx_f+1))
        res_test=np.zeros((1000,num_feature*lp,lx-lx_f+1,ly-lx_f+1))
        k_index=0

        for p in range(lp):
            for feature_mat in kernels:
                for i in range(lx-lx_f+1):
                    for j in range(ly-lx_f+1):
                        res[:,k_index,i,j]=(self.con_mat[:,p,i:i+lx_f,j:j+lx_f]*feature_mat).sum(axis=1).sum(axis=1)
                        res_test[:,k_index,i,j]=(self.con_test[:,p,i:i+lx_f,j:j+lx_f]*feature_mat).sum(axis=1).sum(axis=1)
                k_index+=1
        return res,res_test

    def feature_5_gussian_blur(self):
        return 1.00/256.00*np.array( [ [1 , 6, 4, 6, 1] ,\
                                      [4 , 16, 24, 16, 4], \
                                      [6 , 24, 36, 24, 6], \
                                      [4 , 16, 24, 16, 4],
                                      [1 , 6, 4, 6, 1] ])
    def feature_5_unsharp_masking(self):
        return -1.00/256.00*np.array( [ [1 , 6, 4, 6, 1] ,\
                                      [4 , 16, 24, 16, 4], \
                                      [6 , 24, -476, 24, 6], \
                                      [4 , 16, 24, 16, 4],
                                      [1 , 6, 4, 6, 1] ])
    def feature_5_ones(self):
        return 1.00/25.00*np.ones((5,5))
    def feature_5_gradient1(self):
        return np.array( [ [2 , 2, 2, 2, 2],\
                [1,1,1,1,1],\
                [0,0,0,0,0],\
                [-1,-1,-1,-1,-1] ,\
                [-2,-2,-2,-2,-2] ] )

    def feature_5_gradient2(self):
        return np.array( [ [2 , 1, 0, -1, -2],\
                [2,1,0,-1,-2] ,\
                [2,1,0,-1,-2] ,\
                [2,1,0,-1,-2] ,\
                [2,1,0,-1,-2]  ] )

    def feature_5_sobel1(self):
        return np.array( [ [2 , 2, 4, 2, 2],\
                [1 , 1, 2, 1, 1],\
                [0 , 0, 0, 0, 0],\
                [-1 , -1, -2, -1, -1],\
                [-2 , -2, -4, -2, -2] ] )

    def feature_5_sobel2(self):
        return np.array( [ [2 , 1, 0, -1, -2],\
                [2 , 1, 0, -1, -2],\
                [4 , 1, 0, -1, -4],\
                [2 , 1, 0, -1, -2],\
                [2 , 1, 0, -1, -2]])

    def feature_3_identity(self):
        return np.array( [ [0 , 0, 0] ,\
                                      [0 , 1, 0], \
                                      [0 , 0, 0] ])
    def feature_3_edge(self):
        return np.array( [ [0 , 1, 0] ,\
                                      [1 , -4, 1], \
                                      [0 , 1, 0] ])
    def feature_3_sharpen(self):
        return np.array( [ [0 , -1, 0] ,\
                                      [-1 , 5, -1], \
                                      [0 , -1, 0] ])
    def feature_3_box_blur(self):
        return 1.00/9.00*np.ones((3,3))
    
    def feature_3_guassian_blur(self):
        return 1.00/16.00*np.array( [ [1 , 2, 1] ,\
                                      [2 , 4, 2], \
                                      [1 , 2, 1] ])
    def feature_3_bottom_sobel(self):
        return np.array( [ [ -1,-2,-1], [0, 0,0 ], [ 1,2,1]  ]  )
    def feature_3_emboss(self):
        return np.array( [ [ -2,-1,0], [-1, 1,1 ], [ 0,1,2]  ]  )

    def feature_3_left_sobel(self):
        return np.array( [ [ 1,0,-1], [2, 0,-2 ], [ 1,0,-1]  ]  )

    def feature_3_outline(self):
        return np.array( [ [ -1,-1,-1], [-1, 8,-1 ], [ -1,-1,-1]  ]  )

    def feature_3_right_sobel(self):
        return np.array( [ [ -1,0,1], [-2, 0,2 ], [ -1,0,1]  ]  )

    def feature_3_top_sobel(self):
        return np.array( [ [ 1,2,1], [0, 0,0 ], [ -1,-2,-1]  ]  )

    def feature_3_edge1(self):
        return np.array( [ [ 0,0,0], [-1, 2,-1 ], [ 0,0,0]  ]  )

    def feature_3_edge2(self):
        return np.array( [ [ 0,-1,0], [0, 2,0 ], [ 0,-1,0]  ]  )

    def feature_3_gradient1(self):
        return np.array( [ [ -1,-1,-1], [0, 0,0 ], [ 1,1,1]  ]  )

    def feature_3_gradient2(self):
        return np.array( [ [ -1,0,1], [-1, 0,1 ], [ -1,0,1]  ]  )

    def pooling(self,mat,mat_test,lx_f):
        lx=(np.shape(mat))[2]
        ly=(np.shape(mat))[3]
        lf=(np.shape(mat))[1]
        res=np.zeros((5000,lf,int(lx/lx_f),int(ly/lx_f)))
        res_test=np.zeros((1000,lf,int(lx/lx_f),int(ly/lx_f)))
        for k in range(lf):
            for i in range(int(lx/lx_f)):
                for j in range(int(ly/lx_f)):
                    res[:,k,i,j]= ( (mat[:,k,2*i:2*i+lx_f,2*j:2*j+lx_f]).max(axis=1) ).max(axis=1)
                    res_test[:,k,i,j]= ( (mat_test[:,k,2*i:2*i+lx_f,2*j:2*j+lx_f]).max(axis=1) ).max(axis=1)

        self.con_mat=res
        self.con_test=res_test
    def flatten(self):
        self.train_pixel=self.con_mat.reshape(5000,-1)
        self.test_pixel=self.con_test.reshape(1000,-1)
        self.n_p=(np.shape(self.train_pixel))[1]
        self.num_of_par=self.n_p*self.n_l1+self.n_l1+self.n_l1*self.n_o+self.n_o
        

        self.n_in_l1=self.n_p*self.n_l1
        self.n_in_l_b=self.n_p*self.n_l1+self.n_l1
        self.n_till_o=self.n_p*self.n_l1+self.n_l1+self.n_l1*self.n_o

    def sigmoid(self,inp):
        return 1.00/(1.00+np.exp(-inp))
    def relu(self,inp):
        return abs(inp) * (inp > 0)
    def hsign(self,inp):
        return 1 * (inp > 0)
    def likelihood(self,W,B,J,B_o):
        drop_arr=np.random.binomial(1,1-self.dropout_rate,self.n_l1)
        self.h=self.relu(np.dot(self.train_pixel, W )+B)*drop_arr
        self.hs=self.hsign(np.dot(self.train_pixel, W )+B)*drop_arr

        y_hat=np.dot(self.h , J)+B_o

        #print y_hat,'1'
        maxyhat=y_hat.max(axis=1)
        #print maxyhat,'2'
        maxyhat_re=maxyhat.reshape(-1,1)
        #print np.shape(maxyhat_re),'3'
        maxyhat_re_size_same=np.tile(maxyhat_re,(1,10 ))
        #print np.shape(maxyhat_re_size_same),'4'
        #print y_hat
    
        self.y_hat_scaled_to_zero=y_hat-maxyhat_re_size_same
        #print self.y_hat_scaled_to_zero,'dizz'
        y_hat_exp=np.exp( self.y_hat_scaled_to_zero)
        y_hat_sum=y_hat_exp.sum(axis=1)
        y_hat_sum=y_hat_sum.reshape(-1,1)
        
        y_hat_exp_sum_matr=np.tile( y_hat_sum, (1,10))
    
        soft=y_hat_exp/y_hat_exp_sum_matr
    
        
        loss=self.d_true*np.log(soft)
        #print soft,'soft'
        loss=(loss.sum(axis=1)).sum()
       
        #print -loss,'loss'
        return -loss,self.y_hat_scaled_to_zero#1.00/(len(digits))*np.sum(loss_class)
    
    def log_lik_cal(self ,y_p,J):

 
        ysum=np.exp(y_p).sum(axis=1)
        
        g_w= (np.dot( self.train_pixel.T, (np.dot( -self.d_true+np.exp(self.y_hat_scaled_to_zero)/(ysum.reshape(-1,1)) ,J.T) )*self.hs )).ravel()
 

        g_b= (np.dot( -self.d_true+np.exp(self.y_hat_scaled_to_zero)/(ysum.reshape(-1,1)) , J.T )).sum(axis=0) 

        g_j=( np.dot (self.h.T, (-self.d_true+np.exp(self.y_hat_scaled_to_zero)/(ysum.reshape(-1,1))))  ).ravel()
    
        
        g_o= (-self.d_true+np.exp(self.y_hat_scaled_to_zero)/(ysum.reshape(-1,1))).sum(axis=0)

        return np.concatenate((g_w,g_b,g_j,g_o))
    def optimizer_adam(self,tau):
        sm=[]
        param=[]
        m_hat_s=[]
        v_hat_s=[]
        v_s=[]
        m_s=[]
        self.cost=[]
        beta1=0.9
        alph=1e-8
        beta2=0.999
        for ii in range(self.num_of_par):
            sm.append(np.random.random()*0.001)
            param.append([])
        
        m=np.zeros(len(sm))
        v=np.zeros(len(sm))
        
        arg,y_pro=self.likelihood(np.array(sm)[:self.n_in_l1].reshape(self.n_p,self.n_l1),np.array(sm)[self.n_in_l1:\
                        self.n_in_l_b],np.array(sm)[self.n_in_l_b:self.n_till_o].reshape(self.n_l1,self.n_o),\
                                    np.array(sm)[self.n_till_o:])
        
        
        
        for i in tqdm(range(self.iteration)):


            ans_der=self.log_lik_cal(y_pro,np.array(sm)[self.n_in_l_b:self.n_till_o].reshape(self.n_l1,self.n_o))
            
            m=beta1*m+(1.00-beta1)*ans_der
            v=beta2*v+(1.00-beta2)*ans_der**2
            m_hat=m/(1.00-beta1**(i+1))
            v_hat=v/(1.00-beta2**(i+1))
            
            m_hat_s.append(m_hat)
            v_hat_s.append(v_hat)
            v_s.append(v)
            m_s.append(m)
            
            sm=sm-tau*m_hat/(np.sqrt(v_hat)+alph)

            arg,y_pro=self.likelihood(np.array(sm)[:self.n_in_l1].reshape(self.n_p,self.n_l1),np.array(sm)[self.n_in_l1:\
                        self.n_in_l_b],np.array(sm)[self.n_in_l_b:self.n_till_o].reshape(self.n_l1,self.n_o),\
                                    np.array(sm)[self.n_till_o:])
            
            self.cost.append(arg)
        self.par_mean=np.array(sm)

    def optimizer_gd(self):
        sm=[]
        param=[]
        self.cost=[]

        for ii in range(self.num_of_par):
            sm.append(np.random.random()*0.001)
            param.append([])
        
        m=np.zeros(len(sm))
        v=np.zeros(len(sm))
        
        arg,y_pro=self.likelihood(np.array(sm)[:self.n_in_l1].reshape(self.n_p,self.n_l1),np.array(sm)[self.n_in_l1:\
                        self.n_in_l_b],np.array(sm)[self.n_in_l_b:self.n_till_o].reshape(self.n_l1,self.n_o),\
                                    np.array(sm)[self.n_till_o:])
        
        
        tau=0.000001
        for i in tqdm(range(self.iteration)):

            ans_der=self.log_lik_cal(y_pro,np.array(sm)[self.n_in_l_b:self.n_till_o].reshape(self.n_l1,self.n_o))
            sm=sm-tau*ans_der
            arg,y_pro=self.likelihood(np.array(sm)[:self.n_in_l1].reshape(self.n_p,self.n_l1),np.array(sm)[self.n_in_l1:\
                        self.n_in_l_b],np.array(sm)[self.n_in_l_b:self.n_till_o].reshape(self.n_l1,self.n_o),\
                                    np.array(sm)[self.n_till_o:])
            self.cost.append(arg)
        self.par_mean=np.array(sm)
    def langevin_hastings_optimizer(self,n_iter,sigma):  
        sm=[]
        param=[]
        self.cost=[]


      
        for ii in range(self.num_of_par):
            sm.append(np.random.random()*0.001)
            param.append([])
        
        
        arg,y_pro=self.likelihood(np.array(sm)[:self.n_in_l1].reshape(self.n_p,self.n_l1),np.array(sm)[self.n_in_l1:\
                        self.n_in_l_b],np.array(sm)[self.n_in_l_b:self.n_till_o].reshape(self.n_l1,self.n_o),\
                                    np.array(sm)[self.n_till_o:])
        
        ans_der=self.log_lik_cal(y_pro,np.array(sm)[self.n_in_l_b:self.n_till_o].reshape(self.n_l1,self.n_o))
        tau=0.000001
        for i in tqdm( range(n_iter) ):


            smnew=sm[:]

            
            smnew[:]=sm[:]-tau*ans_der+np.sqrt(2*tau)*np.random.randn(self.num_of_par)
            
            argnew,y_pro_new=self.likelihood(np.array(smnew)[:self.n_in_l1].reshape(self.n_p,self.n_l1),np.array(smnew)[self.n_in_l1:\
                        self.n_in_l_b],np.array(smnew)[self.n_in_l_b:self.n_till_o].reshape(self.n_l1,self.n_o),\
                                    np.array(smnew)[self.n_till_o:])
            
            ans_der_new=self.log_lik_cal(y_pro_new,np.array(smnew)[self.n_in_l_b:self.n_till_o].reshape(self.n_l1,self.n_o))

            p_x_xnew_to_p_xnew_x=np.exp(-1.00/(4.00*tau)*\
                        ( np.dot(np.array(sm)-np.array(smnew)-tau*ans_der_new,np.array(sm)-np.array(smnew)-tau*ans_der_new )\
                            -np.dot(np.array(smnew)-np.array(sm)-tau*ans_der,np.array(smnew)-np.array(sm)-tau*ans_der ) ) )

            min_of_dis=min(1,np.exp(-(argnew-arg)/(sigma))*p_x_xnew_to_p_xnew_x)
        
            dran=random.random()

            if argnew-arg!=0 and argnew-arg!=np.nan and min_of_dis>=dran:
                #print 'accept'
                sm[:]=smnew[:]
                y_pro[:,:]=y_pro_new[:,:]
                arg=argnew
                ans_der[:]=ans_der_new[:]
                for zz in range(self.num_of_par):
                    param[zz].append(sm[zz])
                self.cost.append(arg)
            

        self.par_mean=np.zeros(self.num_of_par)
        for hh in range(self.num_of_par):
            par_arr=np.array( param[hh])
            self.par_mean[hh]=np.mean(par_arr[500:])


    def predict(self):
        
        hj=self.relu( np.dot(self.test_pixel, self.par_mean[:self.n_in_l1].reshape(self.n_p,self.n_l1) )\
                      +self.par_mean[self.n_in_l1:self.n_in_l_b] )
        self.res_predict=np.dot(hj, self.par_mean[self.n_in_l_b:self.n_till_o].reshape(self.n_l1,self.n_o))+self.par_mean[self.n_till_o:]
        return np.argmax( self.res_predict,axis=1)
    def evaluate(self):
        test_javab_ind=np.argmax( self.res_predict,axis=1)
        booli=self.test_label==test_javab_ind
        #print len( booli[booli==True])/np.float(len(booli))  ,'Percentage of True prediction'
        return (len( booli[booli==True])/np.float(len(booli)))
    def plot_loss(self):
        plt.figure(0)
        plt.plot(self.cost,'-.',color='blue',label='adam')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
if __name__ == '__main__':
    iteration=400
    dataname='./train.csv'
    n_l1=50
    n_o=10
    n_p=28*28
    rate_adam=0.001
    
    
    num_rec=number_recognition(iteration,dataname,n_p,n_l1,n_o)
    num_rec.get_data()
    num_rec.get_train_test()
    num_rec.drop_out(0.14)
    t1=timeit.default_timer()
    convr,convtest=num_rec.convolution(5, 5)
    num_rec.pooling(convr,convtest, 2)
    convr,convtest=num_rec.convolution(12, 3)
    num_rec.pooling(convr,convtest, 2)
    num_rec.flatten()
    ###num_rec.langevin_hastings_optimizer(n_iter=3000,sigma=4000)
    #cost_gd=num_rec.optimizer_gd()
    num_rec.optimizer_adam(rate_adam)
    print( "Elapsed Time = "+"{0:.2f}".format( timeit.default_timer()-t1 ) )
    num_rec.predict()
    tru_pr=num_rec.evaluate()
    print( "Accuracy = "+"{0:.2f}".format( tru_pr ) )
    num_rec.plot_loss()
    
