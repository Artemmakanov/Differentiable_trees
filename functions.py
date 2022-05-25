import jax.numpy as jnp
import pandas as pd

class AdamOptim():
    def __init__(self,length_observation, max_depth, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
#         self.m_dw, self.v_dw = 0, 0
#         self.m_db, self.v_db = 0, 0
        
        self.m_dnp, self.v_dnp = \
            jnp.zeros((2**max_depth - 1,  length_observation + 1)), jnp.zeros((2**max_depth - 1,  length_observation + 1))
        self.m_dap, self.v_dap = \
            jnp.zeros((2**max_depth - 1,  2)), jnp.zeros((2**max_depth - 1,  2))
        
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        SimpleOptim
    def update(self,t,np, ap, dnp, dap):
        ## dw, db are from current minibatch
        ## momentum beta 1
        # *** weights *** #
        self.m_dnp = self.beta1*self.m_dnp + (1-self.beta1)*dnp
        # *** biases *** #
        self.m_dap = self.beta1*self.m_dap + (1-self.beta1)*dap

        ## rms beta 2
        # *** weights *** #
        self.v_dnp = self.beta2*self.v_dnp + (1-self.beta2)*jnp.square(dnp)
        # *** biases *** #
        self.v_dap = self.beta2*self.v_dap + (1-self.beta2)*jnp.square(dap)

        ## bias correction
        m_dnp_corr = self.m_dnp/(1-self.beta1**t)
        m_dap_corr = self.m_dap/(1-self.beta1**t)
        v_dnp_corr = self.v_dnp/(1-self.beta2**t)
        v_dap_corr = self.v_dap/(1-self.beta2**t)

        ## update weights and biases
        np = np - self.eta*(m_dnp_corr/(jnp.square(v_dnp_corr).sum()+self.epsilon))
        ap = ap - self.eta*(m_dap_corr/(jnp.square(v_dap_corr).sum()+self.epsilon))
        
        print(self.eta/(jnp.square(v_dnp_corr).sum()+self.epsilon))
#         print()
        return np, ap


class SimpleOptim():
    def __init__(self, max_depth, lr_np, lr_ap, theta):
        
        self.lr_np = lr_np
        self.lr_ap = lr_ap
        self.theta = theta
        
    def update(self,np, ap, dnp, dap, t):

        ## update weights and biases
        np = np - np * self.lr_np / (1 + self.theta * t)
        ap = ap - ap * self.lr_ap / (1 + self.theta * t)
        

        return np, ap



