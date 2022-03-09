from scipy.integrate import solve_ivp
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import animation

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

rcParams.update({'figure.autolayout': True})
rcParams['text.usetex'] = True
#rcParams['text.latex.preamble']=r"\usepackage{bm}"

from func_timoshenko import matrices_timoshenko

init = ('sin(pi*x[0])', 'sin(3*pi*x[0])', '0', '0')

M, J, B, e0, dofs_dict, x_dict = matrices_timoshenko(n_el=20, deg=2, e0_string=init)

dofs_vt = dofs_dict['v_t']
x_vt = x_dict['v_t']

dofs_vr = dofs_dict['v_r']
x_vr = x_dict['v_r']

dofs_sigr = dofs_dict['sig_r']
x_sigr = x_dict['sig_r']

dofs_sigt = dofs_dict['sig_t']
x_sigt = x_dict['sig_t']

u_in = np.array([0, 1])

A_sys = np.linalg.solve(M, J)
B_sys = np.linalg.solve(M, B)

def fun(t,y):

    dydt = A_sys @ y # + B_sys @ u_in * np.sin(pi*t) # * (t<=1 or t>=5) 

    return dydt

t0 = 0.0
t_fin = 10
t_span = [t0, t_fin]

n_ev = 500
t_ev = np.linspace(t0, t_fin, num=n_ev)

sol = solve_ivp(fun, t_span, e0, method='RK45', t_eval = t_ev, \
                       atol = 1e-5, rtol = 1e-5)

e_sol = sol.y

vt_sol= np.zeros((len(dofs_vt), n_ev))
vr_sol= np.zeros((len(dofs_vr), n_ev))

sigr_sol= np.zeros((len(dofs_sigr), n_ev))
sigt_sol= np.zeros((len(dofs_sigt), n_ev))

for i in range(n_ev):
    
    vt_sol[:, i] = e_sol[dofs_vt, i]
    vr_sol[:, i] = e_sol[dofs_vr, i]
    
    sigr_sol[:, i] = e_sol[dofs_sigr, i]
    sigt_sol[:, i] = e_sol[dofs_sigt, i]


H_vec = np.zeros((n_ev))

for i in range(n_ev):
    H_vec[i] = 0.5 *(e_sol[:, i] @ M @ e_sol[:, i])

fig = plt.figure()
plt.plot(t_ev, H_vec, 'g-')
plt.xlabel(r'{Time} [t]')
plt.ylabel(r'Total Energy [J]')

# Plot of the different variables
# Due to fenics ordering, a permutation is first needed

perm_vt = np.argsort(x_vt)
x_vt_perm = x_vt[perm_vt]
vt_sol_perm = vt_sol[perm_vt, :]

perm_vr = np.argsort(x_vr)
x_vr_perm = x_vr[perm_vr]
vr_sol_perm = vr_sol[perm_vr, :]

perm_sigr = np.argsort(x_sigr)
x_sigr_perm = x_sigr[perm_sigr]
sigr_sol_perm = sigr_sol[perm_sigr, :]

perm_sigt = np.argsort(x_sigt)
x_sigt_perm = x_sigt[perm_sigt]
sigt_sol_perm = sigt_sol[perm_sigt, :]

# Plot variables
fig, ax = plt.subplots()
ax.set_xlabel('Space [m]')
ax.set_ylabel('Coenergy variables')
ax.set_xlim(0, 1)
ax.set_ylim(np.min(np.min(e_sol)), np.max(np.max(e_sol)))

line_vt, = ax.plot([], [], lw=2, label = '${v}_t$ at $t$ =' \
                   + '{0:.2f}'.format(t_ev[0]) + '[s]')
line_vr, = ax.plot([], [], lw=2, label = '${v}_r$ at $t$ =' \
                   + '{0:.2f}'.format(t_ev[0]) + '[s]')
line_sigr, = ax.plot([], [], 'o', lw=2, label = '${\sigma}_r$ at $t$ ='  \
                   + '{0:.2f}'.format(t_ev[0]) + '[s]')
line_sigt, = ax.plot([], [], '*', lw=2, label = '${\sigma}_t$ at $t$ ='  \
                   + '{0:.2f}'.format(t_ev[0]) + '[s]')

# Functions for plot
def animate(i):
    line_vt.set_data(np.pad(x_vt_perm, (1, 0)), np.pad(vt_sol_perm[:,i], (1, 0)))
    line_vr.set_data(np.pad(x_vr_perm, (1, 0)), np.pad(vr_sol_perm[:,i], (1, 0)))
    line_sigr.set_data(x_sigr_perm, sigr_sol_perm[:,i])
    line_sigt.set_data(x_sigt_perm, sigt_sol_perm[:,i])
    
    line_vt.set_label('${v}_t$')
    line_vr.set_label('${v}_r$')
    line_sigr.set_label('${\sigma}_r$')
    line_sigt.set_label('${\sigma}_t$')
    
#    line_vt.set_label('$\mathbf{v}_t$ at $t$ =' + '{0:.2f}'.format(t_ev[i]) + '[s]')
#    line_vr.set_label('$\mathbf{v}_r$ at $t$ =' + '{0:.2f}'.format(t_ev[i]) + '[s]')
#    line_sigr.set_label('$\bm{\sigma}_r$ at $t$ =' + '{0:.2f}'.format(t_ev[i]) + '[s]')
#    line_sigt.set_label('$\bm{\sigma}_t$ at $t$ =' + '{0:.2f}'.format(t_ev[i]) + '[s]')
#    
    ax.legend(bbox_to_anchor=(1.25, 1.25))

    return [line_vt, line_vr, line_sigr, line_sigt]


anim = animation.FuncAnimation(fig, animate, frames=len(t_ev), interval=20, blit=False)

##path_out = "/home/andrea/Videos/"
##Writer = animation.writers['ffmpeg']
##writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
##anim.save(path_out + 'timo_bc.mp4', writer=writer)

