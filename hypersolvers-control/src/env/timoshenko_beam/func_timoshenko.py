from fenics import *
import numpy as np
# from scipy import linalg as la
    
def matrices_timoshenko(n_el=10, deg=1, e0_string=('0', '0', '0', '0'),\
                        rho=1, I_rho=1, C_b=1, C_s=1, L=1):
    """
    Computes matrices M, J, B for the Timoshenko beam.
    Parameters:
    n_el: number of finite elements
    deg: degree of  palynomial basis functions
    rho: density per unit length [kg/m]
    I_rho: rotational inertia
    C_b: bending compliance
    C_s: shear compliance
    L: beam length
    """

    # Mesh
    mesh = IntervalMesh(n_el, 0, L)
    d = mesh.geometry().dim()
   
    class AllBoundary(SubDomain):
        """
        Class for defining the two boundaries
        """
        def inside(self, x, on_boundary):
            return on_boundary
    
    
    class Left(SubDomain):
        """
        Class for defining the left boundary
        """
        def inside(self, x, on_boundary):
            return abs(x[0] - 0.0) < DOLFIN_EPS and on_boundary
    
    class Right(SubDomain):
        """
        Class for defining the right boundary
        """
        def inside(self, x, on_boundary):
            return abs(x[0] - L) < DOLFIN_EPS and on_boundary
    
    
    # Boundary conditions on displacement
    all_boundary = AllBoundary()
    # Boundary conditions on rotations
    left = Left()
    right = Right()
    
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    left.mark(boundaries, 1)
    right.mark(boundaries, 2)
   
    # Measures for the evaluation of forms on the domain
    dx = Measure('dx')
    # Measure for evaluating boundary forms
    ds = Measure('ds', subdomain_data=boundaries)
    
    # Finite elements defition{}
    P_pw = FiniteElement('CG', mesh.ufl_cell(), deg)
    P_pth = FiniteElement('CG', mesh.ufl_cell(), deg)
    P_qth = FiniteElement('DG', mesh.ufl_cell(), deg-1)
    P_qw = FiniteElement('DG', mesh.ufl_cell(), deg-1)
    
    # Our problem is defined on a mixed function space
    element = MixedElement([P_pw, P_pth, P_qth, P_qw])
    
    V = FunctionSpace(mesh, element)
    
    v = TestFunction(V)
    v_pw, v_pth, v_qth, v_qw = split(v)

    e = TrialFunction(V)
    e_pw, e_pth, e_qth, e_qw = split(e)
    
    exp_e0 = Expression(e0_string, degree=2)
    e0_array = interpolate(exp_e0, V).vector().get_local()

    # Energy variables
    al_pw = rho * e_pw
    al_pth = I_rho * e_pth
    al_qth = C_b * e_qth
    al_qw = C_s * e_qw
    
    def get_m_form(v_pw, v_pth, v_qth, v_qw, al_pw, al_pth, al_qth, al_qw):
        """
        Defines the mass form. Once assembled the mass matrix is obtained
        """
        m = v_pw * al_pw * dx \
            + v_pth * al_pth * dx \
            + v_qth * al_qth * dx \
            + v_qw* al_qw * dx 
        
        return m


    def get_j_form(v_pw, v_pth, v_qth, v_qw, e_pw, e_pth, e_qth, e_qw):
        """
        Defines the interconnection form.
        Once assembled the interconnection matrix is obtained
        """
        
        j_grad = v_qw* e_pw.dx(0) * dx
        j_gradIP = -v_pw.dx(0) * e_qw * dx
        
        j_Grad = v_qth * e_pth.dx(0) * dx
        j_GradIP = -v_pth.dx(0) * e_qth * dx
        
        j_Id = dot(v_pth, e_qw) * dx
        j_IdIP = -dot(v_qw, e_pth) * dx
        
        j = j_grad + j_gradIP + j_Grad + j_GradIP + j_Id + j_IdIP
                
        return j
    
    # Boundary conditions
    bcs=[]
    
    bc_w = DirichletBC(V.sub(0), Constant(0.0), left)
    bc_th = DirichletBC(V.sub(1), Constant(0.0), left)
    
    bcs.append(bc_w)
    bcs.append(bc_th)
    
    dofs_bc = []
    for bc in bcs:
        for key in bc.get_boundary_values().keys():
            dofs_bc.append(key)
    
    # Matrices assembly
    m_form = get_m_form(v_pw, v_pth, v_qth, v_qw, al_pw, al_pth, al_qth, al_qw)
    j_form = get_j_form(v_pw, v_pth, v_qth, v_qw, e_pw, e_pth, e_qth, e_qw)
    
    J_petsc = assemble(j_form)
    M_petsc = assemble(m_form)
    
    J_mat = J_petsc.array()
    M_mat = M_petsc.array()
    
    BF_vec = assemble(v_pw*ds(2)).get_local().reshape((-1, 1))
    BT_vec = assemble(v_pth*ds(2)).get_local().reshape((-1, 1))
    
    B_mat = np.concatenate((BT_vec, BF_vec), axis=1)
     
    # Dofs and coordinates for subspaces
    dofs2x = V.tabulate_dof_coordinates().reshape((-1, d))

    dofsVpw = V.sub(0).dofmap().dofs()
    dofsVpth = V.sub(1).dofmap().dofs()
    dofsVqth = V.sub(2).dofmap().dofs()
    dofsVqw = V.sub(3).dofmap().dofs()
       
    xVpw = dofs2x[dofsVpw, 0]
    xVpth = dofs2x[dofsVpth, 0]
    xVqth = dofs2x[dofsVqth, 0]
    xVqw = dofs2x[dofsVqw, 0]
    
    dofs2x_bc = np.delete(dofs2x, dofs_bc)
        
    dofsVpw_bc = dofsVpw.copy()
    dofsVpth_bc = dofsVpth.copy()
    dofsVqth_bc = dofsVqth.copy()
    dofsVqw_bc = dofsVqw.copy()
    
    # Eliminate bc dofs form subspaces
    for bc_dof in dofs_bc:
        
        if bc_dof in dofsVpw_bc:
            dofsVpw_bc.remove(bc_dof)
        else:
            dofsVpth_bc.remove(bc_dof)
            
    
    # Recompute the dofs after bc elimination        
    for (i_bc, dof_bc) in enumerate(dofs_bc):
    
        for (ind, dof) in enumerate(dofsVpw_bc):
            if dof > dof_bc-i_bc:
                dofsVpw_bc[ind] +=-1 
    
        for (ind, dof) in enumerate(dofsVpth_bc):
            if dof > dof_bc-i_bc:
                dofsVpth_bc[ind] +=-1 
                
        for (ind, dof) in enumerate(dofsVqth_bc):
            if dof > dof_bc-i_bc:
                dofsVqth_bc[ind] +=-1 
    
        for (ind, dof) in enumerate(dofsVqw_bc):
            if dof > dof_bc-i_bc:
                dofsVqw_bc[ind] +=-1 
        
        
    xVpw_bc = dofs2x_bc[dofsVpw_bc]
    xVpth_bc = dofs2x_bc[dofsVpth_bc]
    xVqth_bc = dofs2x_bc[dofsVqth_bc]
    xVqw_bc = dofs2x_bc[dofsVqw_bc]
    
    # Dictonary containing the dofs
    dofs_dict = {'v_t': dofsVpw_bc, 'v_r': dofsVpth_bc,\
                 'sig_r': dofsVqth_bc, 'sig_t': dofsVqw_bc}
    
    # Dictonary containing the coordinates
    x_dict = {'v_t': xVpw_bc, 'v_r': xVpth_bc,\
              'sig_r': xVqth_bc, 'sig_t': xVqw_bc}
    
    # Eliminate bc dofs from matrices and vectors
    M_red = np.delete(np.delete(M_mat, dofs_bc, axis=0), dofs_bc, axis=1)
    J_red = np.delete(np.delete(J_mat, dofs_bc, axis=0), dofs_bc, axis=1)
    B_red = np.delete(B_mat, dofs_bc, axis=0)
    e0_red = np.delete(e0_array, dofs_bc, axis=0)
    
    
    return M_red, J_red, B_red, e0_red, dofs_dict, x_dict

 
    
    
    
    
    
    
    