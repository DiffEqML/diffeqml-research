"Contains basic utilities for solving ODEs numerically. Superseded by latest `odeint_hybrid` implementation in `torchdyn'"

import torch
import torch.nn as nn
from torchdyn.numerics._constants import *
from typing import List
from torchdyn.numerics.utils import hairer_norm, init_step, adapt_step
from torchdyn.numerics.utils import EventState

from typing import Tuple, Union, List


def odeint_hybrid(f, x, t_span, j_span, solver, callbacks, atol=1e-3, rtol=1e-3, event_tol=1e-4, priority='jump',
				  seminorm:Tuple[bool, Union[int, None]]=(False, None)):
	"""Solve an initial value problem (IVP) determined by function `f` and initial condition `x`, with jump events defined 
	   by a callbacks.
	"""
	# instantiate the solver in case the user has specified preference via a `str` and ensure compatibility of device ~ dtype
	if type(solver) == str: solver = str_to_solver(solver, x.dtype)
	x, t_span = solver.sync_device_dtype(x, t_span)
	x_shape = x.shape
	ckpt_counter, ckpt_flag, jnum = 0, False, 0
	t_eval, t, T = t_span[1:], t_span[:1], t_span[-1]
	
	###### initial jumps ###########
	event_states = EventState([False for _ in range(len(callbacks))])

	if priority == 'jump':
		new_event_states = EventState([cb.check_event(t, x) for cb in callbacks])
		triggered_events = event_states != new_event_states
		# check if any event flag changed from `False` to `True` within last step
		triggered_events = sum([(a_ != b_)  & (b_ == False)
								for a_, b_ in zip(new_event_states.evid, event_states.evid)])
		if triggered_events > 0:
			i = min([i for i, idx in enumerate(new_event_states.evid) if idx == True])
			x = callbacks[i].jump_map(t, x)
			jnum = jnum + 1

	################## initial step size setting ################
	k1 = f(t, x)
	dt = init_step(f, k1, x, t, solver.order, atol, rtol)

	#### init solution & time vector ####
	eval_times, sol = [t], [x]

	while t < T and jnum < j_span:
		
		############### checkpointing ###############################
		if t + dt > t_span[-1]:
			dt = t_span[-1] - t
		if t_eval is not None:
			if (ckpt_counter < len(t_eval)) and (t + dt > t_eval[ckpt_counter]):
				dt_old, ckpt_flag = dt, True
				dt = t_eval[ckpt_counter] - t
				ckpt_counter += 1

		################ step
		f_new, x_new, x_err, _ = solver.step(f, x, t, dt, k1=k1)

		################ callback and events ########################
		new_event_states = EventState([cb.check_event(t + dt, x_new) for cb in callbacks])
		triggered_events = sum([(a_ != b_)  & (b_ == False)
								for a_, b_ in zip(new_event_states.evid, event_states.evid)])


		# if event, close in on switching state in [t, t + Δt] via bisection
		if triggered_events > 0:
			
			dt_pre, t_inner, dt_inner, x_inner, niters = dt, t, dt, x, 0
			max_iters = 100  # TODO (numerics): compute tol as function of tolerances

			while niters < max_iters and event_tol < dt_inner:
				with torch.no_grad():
					dt_inner = dt_inner / 2
					f_new, x_, x_err, _ = solver.step(f, x_inner, t_inner, dt_inner, k1=k1)

					new_event_states = EventState([cb.check_event(t_inner + dt_inner, x_)
												   for cb in callbacks])
					triggered_events = sum([(a_ != b_)  & (b_ == False)
											for a_, b_ in zip(new_event_states.evid, event_states.evid)])
					niters = niters + 1

				if triggered_events == 0: # if no event, advance start point of bisection search
					x_inner = x_
					t_inner = t_inner + dt_inner
					dt_inner = dt
					k1 = f_new
			x = x_inner
			t = t_inner
			i = min([i for i, x in enumerate(new_event_states.evid) if x == True])

			# save state and time BEFORE jump
			sol.append(x.reshape(x_shape))
			eval_times.append(t.reshape(t.shape))

			# apply jump func.
			x = callbacks[i].jump_map(t, x)

			# save state and time AFTER jump
			sol.append(x.reshape(x_shape))
			eval_times.append(t.reshape(t.shape))

			# reset k1
			k1 = None
			dt = dt_pre

		else:
			################# compute error #############################
			if seminorm[0] == True: 
				state_dim = seminorm[1]
				error = x_err[:state_dim]
				error_scaled = error / (atol + rtol * torch.max(x[:state_dim].abs(), x_new[:state_dim].abs()))
			else: 
				error = x_err
				error_scaled = error / (atol + rtol * torch.max(x.abs(), x_new.abs()))
			
			error_ratio = hairer_norm(error_scaled)
			accept_step = error_ratio <= 1

			if accept_step:
				t = t + dt
				x = x_new
				sol.append(x.reshape(x_shape))
				eval_times.append(t.reshape(t.shape))
				k1 = f_new

			if ckpt_flag:
				dt = dt_old - dt
				ckpt_flag = False
			################ stepsize control ###########################
			dt = adapt_step(dt, error_ratio,
							solver.safety,
							solver.min_factor,
							solver.max_factor,
							solver.order)

	return torch.cat(eval_times), torch.stack(sol)