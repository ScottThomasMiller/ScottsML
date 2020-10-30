
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import torch
from ddp.hnn import HNN, BLNN
from ddp.utils import from_pickle, logmsg
import numpy as np


random_seed=62
#random_seed=42976
torch.manual_seed(random_seed)
np.random.seed(random_seed)
colors = ['blue', 'red','green','black','purple','yellow','orange','gray']

#SP:
base_label = "20200127_123451-dsr0.01-dims200_200-lr0.001"
hnn_label = "20200127_125757-dsr0.01-dims200_200-lr0.001"

def load_hnn_model(path):
    logmsg("loading HNN model {}".format(path))
    saved_model = from_pickle(path)
    args = saved_model['args']
    model = HNN(d_in=3, d_hidden=args.hidden_dim, d_out=1, activation_fn=args.activation_fn)
    model.load_state_dict(saved_model['model'])
    model.eval()

    return (args, model)
    
def load_base_model(path):
    logmsg("loading base model {}".format(path))
    saved_model = from_pickle(path)
    args = saved_model['args']
    model = BLNN(d_in=3, d_hidden=args.hidden_dim, d_out=3, activation_fn=args.activation_fn)
    #model.load_state_dict(torch.load(path,map_location=torch.device('cpu')),strict=False)
    model.load_state_dict(saved_model['model'])
    model.eval()

    return (args, model)
    
def load_stats(base, max, verbose=False):
    stats = []
    for i in range(4):
        fname = "{}-gpu{}.trch".format(base, i)
        stats.append(from_pickle(fname))
        if verbose:
            logmsg("stats {}: {} file".format(len(stats[i]['training']), fname))
        stats[i]['training'] = stats[i]['training'][:max]
        stats[i]['testing'] = stats[i]['testing'][:max]
        stats[i]['grad_norms'] = stats[i]['grad_norms'][:max]
        stats[i]['grad_stds'] = stats[i]['grad_stds'][:max]
    
    return stats

def model_update(t, state, model):
    ''' Procedure model_update is for integrating a vector field parameterized by a NN or HNN.
        It is used by the DynamicalSystem.get_orbit method to calculate the derivatives via the model.'''
    q = state[0]
    p = state[1]    
    x = np.cos(q)
    y = np.sin(q)
    state = np.column_stack((x,y,p))
    state = state.reshape(-1,3)
    deriv = np.zeros_like(state, dtype=np.float32)
    np_x = state 
    x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32)
    dx_hat = model.time_derivative(x)
    deriv = dx_hat.detach().data.numpy()
    ydot = deriv[0,1]
    pdot = deriv[0,2]
    qdot = ydot / np.cos(q)
    qderiv = np.column_stack((qdot, pdot))
    
    return qderiv.reshape(-1)

from ddp.spdata import SimplePendulumDS
import math 

def gen_dynsys(args):
    state_symbols = ['q','p']
    tpoints = abs(int((1.0 / args.dsr) * (args.tspan[1]-args.tspan[0])))
    spsys = SimplePendulumDS(sys_hamiltonian=args.hamiltonian,state_symbols=state_symbols,
                          tspan=args.tspan, timesteps=tpoints, integrator=args.integrator_scheme,
                          symplectic_order=4)
    spsys.integrator = "RK45"

    return spsys


def load_models():
    base_args, base_model, hnn_args, hnn_model = (None, None, None, None)
    base_path = "ddp/baseline_model_{}-gpu-avg.trch".format(base_label)
    hnn_path = "ddp/hnn_model_{}-gpu-avg.trch".format(hnn_label)
    base_args, base_model = load_base_model(base_path)
    hnn_args, hnn_model = load_hnn_model(hnn_path)
    models = (base_args, base_model, hnn_args, hnn_model)
    
    return models

def gen_orbits(spsys, base_model, hnn_model):
    save_state = spsys.random_config()
    save_state[0] = 2.92950555
    save_state[1] = 0.06535495
    #[2.92950555 0.06535495]
    
    logmsg("save state: {}".format(save_state))
    base_orbit, base_settings = (None, None)
    if base_model is not None:
        base_model.eval()
    hnn_model.eval()
    
    logmsg("calculating HNN orbit")
    update_fn = lambda t, y0: model_update(t, y0, hnn_model)
    spsys.external_update_fn = update_fn
    state = save_state
    hnn_orbit, hnn_settings = spsys.get_orbit(state)
    logmsg("hnn shape: {}".format(hnn_orbit.shape))

    logmsg("calculating ground truth orbit")
    spsys.external_update_fn = None
    state = save_state
    ground_orbit, ground_settings = spsys.get_orbit(state)
    logmsg("ground shape: {}".format(ground_orbit.shape))

    logmsg("calculating baseline orbit")
    update_fn = lambda t, y0: model_update(t, y0, base_model)
    spsys.external_update_fn = update_fn
    state = save_state
    base_orbit, base_settings = spsys.get_orbit(state)
    logmsg("base shape: {}".format(base_orbit.shape))
    
    orbits = (ground_orbit, ground_settings, base_orbit, base_settings, hnn_orbit, hnn_settings)
    return orbits

def plot_energy(data, orbits, spsys, models):
    ground_orbit, ground_settings, base_orbit, base_settings, hnn_orbit, hnn_settings = orbits
    base_args, base_model, hnn_args, hnn_model = models
    fig = plt.figure(figsize=[15,4], dpi=100, linewidth=1)
    ymin = 0
    ymax = 10

    plt.subplot(1,3,1)
    plt.title('Ground Truth')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    xtime = ground_settings['t_eval']
    yenergy = spsys.get_energy(ground_orbit)
    plt.plot(xtime, yenergy, 'k-', c=colors[0], label='true energy', linewidth=1)  
    plt.ylim(ymin, ymax)  
    
    plt.subplot(1,3,2)
    plt.title('NN')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    xtime = base_settings['t_eval'] 
    yenergy = spsys.get_energy(base_orbit)
    plt.plot(xtime, yenergy, 'k-', c=colors[0], label='baseline energy', linewidth=1)  
    plt.ylim(ymin, ymax) 
    
    plt.subplot(1,3,3)
    plt.title('HNN')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    xtime = hnn_settings['t_eval']
    yenergy = spsys.get_energy(hnn_orbit)
    plt.plot(xtime, yenergy, 'k-', c=colors[0], label='HNN energy', linewidth=1)  
    plt.ylim(ymin, ymax)    
    
    plt.tight_layout()
    fig.savefig("ddp/sp_energies-{}.png".format(random_seed))
    plt.show()


def plot_energy_surface(model, name):
    q = np.arange(-math.pi, math.pi, 0.01)
    p = np.arange(-math.pi, math.pi, 0.01)
    Q, P = np.meshgrid(q, p)
    H = np.zeros_like(Q)
    Gr = np.zeros_like(Q)
    Res = np.zeros_like(Q)
    Gr00 = spsys.get_energy([0,0])
    x00 = np.cos([0])
    y00 = np.sin([0])
    state = np.column_stack((x00,y00,0.0))
    o = torch.tensor(state, requires_grad=True, dtype=torch.float32)
    h00, = model.forward(o)
    H00 = h00.detach().data.numpy()
    shift = H00 - Gr00
    fout = open("simple-qpH.txt", "w")
    for i in range(len(q)):
        x = np.cos(q[i])
        y = np.sin(q[i])
        for j in range(len(p)):
            state = np.column_stack((x,y,p[j]))
            state = state.reshape(-1,3)
            o = torch.tensor(state, requires_grad=True, dtype=torch.float32)
            h, = model.forward(o)
            print("h: {}".format(h)) if (i==0 and j==0) else None
            H[i,j] = h.detach().data.numpy()           # HNN output
            Gr[i,j] = spsys.get_energy([q[i], p[j]])   # Ground truth
            Res[i,j] = Gr[i,j] - H[i,j] + shift        # Residual
            fout.write("{}\t{}\t{}\n".format(q[i], p[j], H[i,j]))
            
    fout.close()
    fig = plt.figure(figsize=[10,10], dpi=100, linewidth=1)
    ax = fig.gca(projection='3d')
    #surf = ax.plot_surface(Q, P, Res, cmap=cm.Spectral, linewidth=0, antialiased=False)
    surf = ax.plot_surface(Q, P, H, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('q')
    ax.set_ylabel('p')
    ax.set_zlabel('energy')
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig("ddp/sp_energy_surface_{}.png".format(name))
    plt.show()



# main
models = load_models()
base_args, base_model, hnn_args, hnn_model = models
#base_args.tspan = [0, 1000]
hnn_args.tspan = [0, 100]
spsys = gen_dynsys(hnn_args)
input_data =  spsys.get_dataset(hnn_args.name+".pkl", hnn_args.save_dir)
orbits = gen_orbits(spsys, base_model, hnn_model)
plot_energy(input_data, orbits, spsys, models)
plot_energy_surface(hnn_model, "HNN")
plot_energy_surface(base_model, "NN")
logmsg("done")

