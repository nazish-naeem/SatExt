import numpy as np
import torch
#from models.stitch import StitchRAM
#from models.interpolation import masked_time_average



def generate_land_decomposition(abus, cmask, n_ranks = 3):
    nt, ny, nx, nb = abus.shape
    assert (n_ranks <= nt*nb), "The product of band size and number of samples <= rank" 
    
    #cloud_comp = CloudCompletion()
    #cloud_comp_1 = CloudCompletion(device = torch.device('cuda:0'))
    #cloud_comp_1.init_model(y = abus[:,:10000,:,:].transpose([3,0,1,2]) / 255., 
    #                  mask = cmask[:, :10000:,:,:].transpose([3,0,1,2]),  
    #                  rank = n_ranks, damping_coefficient = 0.01)
    #pred = cloud_comp_1.predict(max_iter = 401, report_freq=50)
    
    if ny * nx * nb >= 60000000 :
        
        ny_ = int(ny / 2)
        cloud_comp_1 = CloudCompletion(device = torch.device('cuda:0'))
        cloud_comp_1.init_model(y = abus[:, :ny_, :,:].transpose([3,0,1,2]) / 255., 
                      mask = cmask[:, :ny_, :,:].transpose([3,0,1,2]),  
                      rank = n_ranks, damping_coefficient = 0.01)
        pred = cloud_comp_1.predict(max_iter = 401, report_freq=50)
        
        cloud_comp_2 = CloudCompletion(device = torch.device('cuda:1'))
        cloud_comp_2.init_model(y = abus[:, ny_:,:,:].transpose([3,0,1,2]) / 255., 
                      mask = cmask[:, ny_:,:,:].transpose([3,0,1,2]),  
                      rank = n_ranks, damping_coefficient = 0.01)
        pred = cloud_comp_2.predict(max_iter = 401, report_freq = 50)
        print("We have two cloud-completion models!!!")
        result = [cloud_comp_1, cloud_comp_2]
    else:
        cloud_comp_1 = CloudCompletion(device = torch.device('cuda:0'))
        cloud_comp_1.init_model(y = abus[:,:,:,:].transpose([3,0,1,2]) / 255., 
                      mask = cmask[:,:,:,:].transpose([3,0,1,2]),  
                      rank = n_ranks, damping_coefficient = 0.01)
        pred = cloud_comp_1.predict(max_iter = 401, report_freq = 50)
        print("We have one cloud-completion model!!!")
        result = [cloud_comp_1]
    return result



def get_land_structure(cloud_models):
    
    if len(cloud_models) == 2:
        land_structure = [cloud_models[0].V.cpu().numpy(),
                                       cloud_models[1].V.cpu().numpy()]
    elif len(cloud_models) == 1:
        land_structure = [cloud_models[0].V.cpu().numpy()]
        
    return land_structure


def get_land_evolution(cloud_models, abus_):
    
    nt, ny, _, nb = abus_.shape
    
    
    if len(cloud_models) == 2:
        ny_ = int(ny /2)
        land_evolution = [cloud_models[0].infer_U(np.transpose(abus_[:,:ny_,:,:], [3,0,1,2]) / 255.).cpu().numpy(),
                          cloud_comp_models[1].infer_U(np.transpose(abus_[:,ny_:,:,:], [3,0,1,2]) / 255.).cpu().numpy()]
    elif len(cloud_models) == 1 :
        land_evolution = [cloud_models[0].infer_U(np.transpose(abus_[:,:,:,:], [3,0,1,2]) / 255.).cpu().numpy()]
        
    return land_evolution




def get_cloud_free_reconstruction(cloud_models, abus_):
    
    nt, height, width, nb = abus_.shape
    land_evolution = get_land_evolution(cloud_models, abus_)
    land_structure = get_land_structure(cloud_models)
    
    print(len(land_evolution), len(land_structure), nb, nt)
    
    if len(cloud_models) == 2:
        image0 = np.matmul(land_evolution[0], np.transpose(land_structure[0], [1,0])).reshape([nt, -1, width, nb])
        image1 = np.matmul(land_evolution[1], np.transpose(land_structure[1], [1,0])).reshape([nt, -1, width, nb])
        image = np.concatenate((image0, image1), axis = 1)
        
        print(image0.shape, image1.shape)
        structure = np.concatenate((land_structure[0], land_structure[1]), axis = 0)
    elif len(cloud_models) == 1:
        image = np.matmul(land_evolution[0], np.transpose(land_structure[0], [1,0])).reshape([nb, nt, -1, width])
        structure = land_structure[0]
    
    n_ranks = np.shape(land_structure[0])[1]
    
    return image, structure.reshape([-1, width, n_ranks])





def delta_matrix(nb, nt):
    """
    Return a matrix Delta such that Delta * X.flat[:] yields a vector
    whose elements are X[b,t+1]-X[b,t] for t=1,..., nt-1 and 0 for t=nt
    when X is an nb x nt matrix.
    
    e.g. we can try this for the following data
    >>> X = np.array([[1,2,3], [4,5,6]])
    >>> D = delta_matrix(2,3)
    >>> (D @ X.flat).reshape(2,3)
    
    array([[1., 1., 0.],
           [1., 1., 0.]])
    """
    d = np.zeros((nt,nt),dtype = np.float32)
    for i in range(nt-1):
        d[i,i] = -1
        d[i,i+1] = 1
    delta = np.kron(np.eye(nb,dtype=np.float32), d)
    return(delta)


def damping_matrix(nb, nt, alpha):
    """
    Calculate the matrix (I+alpha*Delta^T * Delta)^{-1} efficiently,
    where the matrix Delta = delta_matrix(1,nt) kron I_nb
    """
    delta = delta_matrix(1,nt)
    D = alpha * delta.T @ delta
    D.flat[::nt+1] += 1 # adding identity matrix
    D = np.linalg.inv(D)
    D = np.kron(np.eye(nb, dtype=np.float32), D)
    return(D)


def initialize_matrix(n, nr):
    """
    Initialize the matrix such that two matrices initialized with the 
    same routine X Y^T has entries roughly in the range (-1, 1).
    """
    X = np.random.randn(n,nr).astype(np.float32) * (0.2 / np.sqrt(nr))
    return(X)


def masked_time_average(abus, cloud_label):
    """
    This function is like np.mean(real_s2,np.axis=1) if there are no clouds.  When we 
    have clouds we want the same, but without using the cloudy pixels.
    """
    stats = np.sum(abus[:,:,:,:] * (cloud_label[0,...]==0)[np.newaxis,...],axis=1)
    counts = np.sum(cloud_label[0,...] == 0 , axis=0)
    counts[counts==0] = 1 # avoid divide by zero
    pixel_average = stats/counts[np.newaxis]
    return(pixel_average)



class CloudCompletion:
    
    def __init__(self, device = "cpu"):

        # prepare for using torch
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                device = "cpu"
        self.device = torch.device(device)
        print("Using device %s" % device)
        self.damping_coefficient = None # to signal that the model has not been initialized
        #self.x = x
        
    def init_model(self, y, mask, rank = 5, damping_coefficient = 0.5):
        """
        Use a larger rank for more complex or larger scenes and a larger damping_coefficient when
        the cloud mask is not so good.  Otherwise the initial choices should be decent.  
        """
        nb, nt, nx, ny = y.shape
        nr = rank
        alpha = damping_coefficient
        
        #nb1, _, _, _ = self.real_s1.shape
        #nb = nb2
        #if use_s1:
        #    nb = nb1+nb2
        torch.cuda.empty_cache()
        assert(nr < nt*nb and nr < nx*ny) # if rank is larger matrices will become singular
        
        self.damping_coefficient = alpha
        self.damping_matrix = torch.from_numpy(damping_matrix(nb, nt, alpha)).float().to(self.device)
        self.sizes = (nb, nt, nx, ny, nr)
        
        self.cloud_label = mask.astype(int)
        
        self.Y = y.copy()
        
        pixel_average = masked_time_average(y, self.cloud_label)
        for day_num in range(nt):
            cmask = self.cloud_label[0,day_num,:,:]!= 0 
            self.Y[:, day_num, cmask] = pixel_average[:,cmask]
            
        self.cloud_mask = np.ones((nb,nt,nx,ny), np.bool)
        self.cloud_mask[:nb,self.cloud_label[0,:,:,:] == 0] = False
        
        self.Y = self.Y.reshape(nb*nt, nx*ny)
        self.cloud_mask.resize((nb*nt, nx*ny))
        
        self.Y = torch.from_numpy(self.Y).float().to(self.device)
        if hasattr(torch,'bool'): # later versions of torch doesn't like indexing with uint8
            self.cloud_mask = torch.from_numpy(self.cloud_mask.astype(np.bool)).to(self.device)
        else:
            self.cloud_mask = torch.from_numpy(self.cloud_mask.astype(np.uint8)).to(self.device)
        self.U = torch.from_numpy(initialize_matrix(nb*nt, nr)).float().to(self.device)
        self.V = torch.from_numpy(initialize_matrix(nx*ny, nr)).float().to(self.device)
        
        del cmask, pixel_average, y, mask
        torch.cuda.empty_cache()

    def mem_info(self):
        print("GPU MEM = %.2fMB max = %.2fMB cache = %.2fMB" %
              (torch.cuda.memory_allocated(self.device)/1024**2, 
               torch.cuda.memory_cached(self.device) / 1024 ** 2,
               torch.cuda.max_memory_allocated(self.device) / 1024 ** 2))
        
    def update_Z(self):
        """
        Do the update of Z (insize of Y):
        Y[cloud_mask] = (U * V.T)[cloud_mask] 
        """
        nb, nt, nx, ny, nr = self.sizes
        for b in range(nb): # do the matrix multiplication per band to save memory
            Ub = self.U.reshape((nb,nt,nr))[b,:,:].reshape((1*nt, nr))
            Yb = self.Y.reshape((nb, nt, nx, ny))[b,:,:,:].reshape((1*nt,nx*ny))
            #print(Ub.shape, self.V.shape, torch.transpose(self.V, 0, 1).shape)
            Xb = Ub @ torch.transpose(self.V, 0, 1) #(self.V).T
            cm = self.cloud_mask.reshape((nb, nt, nx, ny))[b,:,:,:].reshape((1*nt,nx*ny))
            #print(Xb.type(), Yb.type())
            Yb[cm] = Xb[cm]
            #Yb = Xb   #Yb[cm] = Xb[cm] # update cloudy pixels to get Y_Z
        ## we split up the matrix computation to save memory.  This is the more straightforward code: ...
        X = self.U @  torch.transpose(self.V, 0, 1)  #self.V.T
        self.Y[self.cloud_mask] = X[self.cloud_mask] # Y is now Y_Z
        # Clean
        del X
        torch.cuda.empty_cache()
        
    def update_U(self):
        """
        Update U using the formula:
            U = damping_matrix * Y_Z * V * (V^T * V)^{-1}
        Note that V^T * V is an nr x nr matrix, so inversion is possible as long as we keep nr small.
        """
        #print("Updating U")
        # calculate pseudo inverse of V
        PV = self.V @ torch.inverse(torch.transpose(self.V, 0,1) @ self.V)
        # since nr<nb*nt this is the most efficient order of multiplication
        #print((self.damping_matrix.type()), type(self.Y.type()), type(PV.type()))
        self.U = self.damping_matrix @ (self.Y @ PV) 
        # Clean
        del PV
        torch.cuda.empty_cache()
        
    
    def update_V(self):
        """
        Update V using the formula
            V = Y^T * U * (U^T * U + U^T * Delta * Delta * U )^{-1}
        where Delta is the matrix that computes the forward time difference.  We don't form Delta
        explicitly, but do the matrix multiple implicitly.
        """
        #print("Updating V")
        nb, nt, nx, ny, nr = self.sizes
        alpha = self.damping_coefficient
        DeltaU = 0.* self.U
        DeltaU.reshape(nb,nt,nr)[:,0:nt-1,:] = self.U.reshape(nb,nt,nr)[:,1:nt,:]-self.U.reshape(nb,nt,nr)[:,0:nt-1,:]
        self.V = torch.transpose(self.Y,0,1) @ (self.U @ torch.inverse(torch.transpose(self.U, 0, 1) @ self.U + alpha * DeltaU.transpose(0,1) @ DeltaU))
        #### Clean
        del DeltaU
        torch.cuda.empty_cache()
        
    
    
    def infer_U(self, x):
        nb, _, nx, ny, nr = self.sizes
        nt = x.shape[1]
        X = torch.from_numpy(x.copy().reshape(nb*nt, nx*ny)).float().to(self.device)
        PV = self.V @ torch.inverse(torch.transpose(self.V, 0,1) @ self.V)
        test_damping_matrix = torch.from_numpy(damping_matrix(nb, nt, self.damping_coefficient)).float().to(self.device)
        U = test_damping_matrix @ (X @ PV)
        #pred_X = self.U @ torch.transpose(self.V,0,1)
        #pred_X = pred_X.reshape((nb,nt,nx,ny))[:nb,:,:,:].cpu().numpy()
        return U
    
    def balancing_step(self):
            
        # Note: torch.qr was so slow it is faster to move the data and use numpy
        # for certain sizes torch.qr just hangs forever...
        Qu, Ru = np.linalg.qr(self.U.cpu().numpy())
        Qv, Rv = np.linalg.qr(self.V.cpu().numpy())
        C = Ru @ Rv.T
        Uc, s, Vc = np.linalg.svd(C)
        dc = np.sqrt(np.abs(s))
        self.U = torch.from_numpy(Qu @ Uc @ np.diag(dc)).to(self.device)
        V1 = np.diag(s/dc) @ Vc @ Qv.T
        self.V = torch.from_numpy(V1.T).to(self.device)
        
        
    def completion_loss(self):
        """
        Compute the objective function
        1/2 *||P_{\Omega}(X-Y)||_F^2 + alpha/2 \sum_{t=1}^{T-1} ||X_{t+1}-X_t||_F^2,
        where X = U* V^\top and Omega is the cloud mask.
        """
        nb, nt, nx, ny, nr = self.sizes
        alpha = self.damping_coefficient
        loss = 0.0
        for b in range(nb):
            Ub = self.U.reshape((nb,nt,nr))[b,:,:].reshape((1*nt, nr))
            Yb = self.Y.reshape((nb, nt, nx, ny))[b,:,:,:].reshape((1*nt,nx*ny))
            Xb = Ub @ torch.transpose(self.V, 0,1)
            loss += 0.5 * torch.norm(Xb-Yb)**2
            loss += 0.5 * alpha * torch.norm(Xb.reshape((1, nt, nx, ny))[:,1:,:,:]-Xb.reshape((1,nt,nx,ny))[:,:-1,:,:])**2        
        loss /= (nb*nt*nx*ny) # per pixel loss
        torch.cuda.empty_cache() # Clean
        return(loss)
    
    
    def update_all(self, double_z_update = True):
        # step 1 update Z (inplace in Y)
        self.update_Z()
        # Step 2 update V
        self.update_U()    
        if double_z_update:
            self.update_Z()
        # Step 3 update V
        self.update_V()
        
        
    def predict(self, max_iter=701, report_freq=50, verbose=True):
        """
        This function does the full optimization.  500 iterations is a lot use 200 if you want to run faster.
        """
        cycle_len = 10 # number of steps between each balancing step
        if self.damping_coefficient is None:
            raise(ValueError("obj.init_model(...) needs to be called before predict."))
        with torch.no_grad():
            for it in range(max_iter):
                self.update_all(double_z_update = False)
                if it%report_freq == 0:
                    loss = self.completion_loss()
                    if verbose:
                        print("iter = %d, loss = %10.6f" %(it, float(loss)))
                    if (it == 0) or (loss < self.best_loss):
                        self.best_loss = loss
                        self.best_U = self.U
                        self.best_V = self.V
                if ((it+1) % cycle_len)==0:
                    self.balancing_step()
            self.balancing_step()
            X = self.U @ torch.transpose(self.V,0,1)
            nb,nt,nx,ny,_ = self.sizes
            self.fake_s2 = X.reshape((nb,nt,nx,ny))[:nb,:,:,:].cpu().numpy()
            del X
            torch.cuda.empty_cache() # let someone else have some fun
        return(self.fake_s2)
    
    
    def release_gpu(self):
        del self.U
        del self.V
        del self.Y
        del self.damping_matrix
        del self.cloud_mask
        self.damping_coefficient = None