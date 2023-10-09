import gym
from gym import spaces

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy.special import gammaln
from jax import random

import tifffile
import os
import datetime

##%%

factorial = lambda x: jnp.exp(gammaln(x+1))

class lifeEnv(gym.Env):

    def __init__(self,
                 dim=256,
                 psf_dim=11,
                 pattern=None,
                 NA=1.,
                 wavelength=488,
                 density=.7,
                 ise_0=10,
                 bleach_fudge=.02,
                 noise_mean=.1):
        # initialise self.stuff
        self.key = jax.random.PRNGKey(0)
        self.dim = dim
        self.psf_dim = psf_dim
        self.pattern = pattern
        self.NA = NA
        self.wavelength = wavelength
        self.density = density
        self.t = 0
        self.ise_0 = ise_0
        self.fudge = bleach_fudge
        self.noise_mean = noise_mean
        # create kernel
        self.kernel = jnp.ones((1,1,3,3))
        self.kernel = self.kernel.at[0,0,1,1].set(0)
        # create blur kernel
        l = jnp.linspace(-3,3,7)
        self.blur = jsp.stats.norm.pdf(l) * jsp.stats.norm.pdf(l[:,None])
        self.blur = self.blur[jnp.newaxis,jnp.newaxis,:,:]
        
        #self.binom = lambda x,y: jnp.round(jnp.exp(gammaln(x+1)-gammaln(y+1)-gammaln(x-y+1)))
    
    def radial(m,n,rho):
        
        R = jnp.zeros_like(rho)
        nm = (n-m)//2
        for k in range(nm):
            R += (rho**(n-2*k)) * (-1)**k * factorial(n-k) / \
                (factorial(k) * factorial((n+m)//2 - k) * factorial(nm-k))
        R = jnp.less_equal(rho,1) * R
        return R

    def Zernike(m,n,rho,phi):
        if (n-m)%2 != 0:
            # odd n-m: Z = 0
            Z = jnp.zeros_like(rho)
        else:
            if n-m == 0:
                # n-m = 0: R = rho**n
                R = jnp.ones_like(rho) * rho**n
            else:
                # otherwise, full polynomial
                R = jnp.zeros_like(rho)
                nm = (n-m)//2
                for k in range(nm):
                    R += (rho**(n-2*k)) * (-1)**k * factorial(n-k) / \
                        (factorial(k) * factorial((n+m)//2 - k) * factorial(nm-k))
                R = jnp.less_equal(rho,1) * R
            # introduce phase depending on whether the polynimial is odd
            if m<0:
                Z = R*jnp.sin(m*phi)
            else:
                Z = R*jnp.cos(m*phi)
            # Z.shape == [dim,dim] from rho
        Z = .5*Z * jnp.less_equal(rho,1)
        return Z
    
    def _create_psf(self):
        # convert meshgrid of X,Y to Rho, Phi
        x = jnp.linspace(-1,1,self.dim) # use psf_dim
        y = jnp.linspace(-1,1,self.dim)
        X,Y = jnp.meshgrid(x,y)

        rho = jnp.sqrt((X**2 + Y**2))
        phi = jnp.arctan2(Y,X)
        # evaluate each pixel as sum of c_i*Z(m_i,n_i,rho,phi)
        # use a dictionary {(m,n):c}

        kx = jnp.fft.fftfreq(self.psf_dim)

        pass

    def _create_state(self):
        if self.pattern is None:
            self.pattern = jax.random.uniform(self.key,
                                                shape=(self.dim//2,self.dim//2),
                                                dtype=jnp.float32)
            # create a binary array if value in self.state > .7
            self.pattern = jnp.float32(jnp.greater(self.pattern,self.density))
        else:
            self.pattern = jnp.array([[1 if char == 'O' else 0 for char in row] for row in self.pattern],dtype=jnp.float32)
        x = self.dim//2 - self.pattern.shape[0]//2
        y = self.dim//2 - self.pattern.shape[1]//2
        self.state = jnp.zeros((self.dim,self.dim),dtype=jnp.float32)
        self.state = self.state.at[x:x+self.pattern.shape[0],y:y+self.pattern.shape[1]].set(self.pattern)
        self.state = self.state[jnp.newaxis,jnp.newaxis,:,:]
    
    def _life_update(self):
        state = jnp.float32(self.state)
        neighbours = jax.lax.conv(state,self.kernel,(1,1),'SAME')
        survive = jnp.logical_or(neighbours==3,neighbours==2)
        birth = jnp.equal(neighbours,3)
        state_new = jnp.logical_and(state,survive)
        state_new = jnp.logical_or(state_new,birth)
        self.state = state_new
    
    def _apply_psf(self):
        # generate random intensity by multiplying cells by random values
        state = jnp.float32(self.state)
        # self.gt = f(self.state)
        self.gt = jax.lax.conv(state,self.blur,(1,1),'SAME')
        self.gt = self.gt / self.gt.max()
    
    def _bleach(self):
        ise = self.ise_0 * jnp.exp(-self.fudge*self.exp_sum)
        self.exp_sum += 1.

        self.key, *subkeys = random.split(self.key,3)

        # generate poisson image from self.gt
        obs = jax.random.poisson(subkeys[0],
                                 self.gt * ise)
        obs = jnp.float32(obs)

        n = jax.random.normal(subkeys[1],
                              shape=(1,1,self.dim,self.dim),
                              dtype=jnp.float32)

        self.obs = jnp.clip(obs + n*self.noise_mean,0,None)
        self.obs = self.obs / self.obs.max()
    
    def seed(self,seed):
        self.key = jax.random.PRNGKey(seed)

    def reset(self,seed=None,options=None):
        # create underlying cellular automata img
        self._create_state()
        # alternatively, import from an image
        # set noise timer to 0
        self.t = 0
        self.exp_sum = 0.
        # step
        #return self.step(None)

    def step(self,action):
        # convolve img with ca kernel to update state
        self._life_update()
        # apply psf
        self._apply_psf()
        # apply noise
        self._bleach()
        # return observation
        return self.obs
    
    def render(self,num_frames=100,start=0):
        self.reset()
        cpu = jax.devices('cpu')[0]
        s_list = []
        g_list = []
        o_list = []
        for _ in range(start):
            _ = self.step(None)
        for _ in range(num_frames):
            # step forward
            obs = self.step(None)
            s_list.append(jax.device_put(self.state[0,0],device=cpu))
            g_list.append(jax.device_put(self.gt[0,0],device=cpu))
            o_list.append(jax.device_put(obs[0,0],device=cpu))

        s_all = jnp.stack(s_list)
        g_all = jnp.stack(g_list)
        o_all = jnp.stack(o_list)

        name = datetime.datetime.now().strftime('sim%Y%m%dT%H%M')
        os.mkdir(name)

        tifffile.imwrite(name+'/s_all.tif',s_all)
        tifffile.imwrite(name+'/g_all.tif',g_all)
        tifffile.imwrite(name+'/o_all.tif',o_all)
