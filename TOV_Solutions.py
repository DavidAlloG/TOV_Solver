"""
Integrate TOV equations for Neutron Stars Equation of State

David García Allo
June 2022

Reference: https://github.com/dlwhittenbury/PYTOV
"""

#Import Modules
import matplotlib.pyplot as plt
import numpy as np

#Constants for units changing
MeVfm_3Tokm_2 = 1.3234e-6
km_2ToMeVfm_3 = 1.0 / MeVfm_3Tokm_2
MsunTokm = 1.4766
kmToMsun = 1.0 / MsunTokm

"""----------------------------------------------------------------------------    
                            Creating the funcions needed
----------------------------------------------------------------------------"""
def read_EoS(filename):
    """Read the file located in the same directory
    Columns: 0->density; 1->pressure; 2->energy density"""
    return np.loadtxt(filename)

def join_EoS(low_density_EoS, high_density_EoS):
    """Join the low and high density equation of state to have a combined
    equation of state in teh full range of density"""
    #Check when the index where the values of low EoS are higher than high EoS
    index1 = np.argmax(low_density_EoS[:, 0] > high_density_EoS[0, 0])
    index2 = np.argmax(low_density_EoS[:, 1] > high_density_EoS[0, 1])
    index3 = np.argmax(low_density_EoS[:, 2] > high_density_EoS[0, 2])
    #We keep the minimun index of the last 3
    index = min(index1,index2, index3)
    #Cut the low density EoS till that index
    low_density_EoS_cut = low_density_EoS[:index, :]
    #Return the combined EoS
    return np.row_stack((low_density_EoS_cut, high_density_EoS))

def density(pressure):
    """Returns density [fm-3] as a function of pressure [km-2]
    Only interpolation, no extrapolation"""
    if pressure < 0:
        return 0 #No negative pressures
    else:
        #Firs index where the pressure of the EoS is higher than actual pressure
        i = np.argmax(pressure_EoS > pressure)
        if i == 0: #If is is the first index we dont extrapolate to lower values
            return density_EoS[0]
        else:
            last_i = len(density_EoS) - 1
            if i <= last_i: #Interpolate if 0 < i <= last_index
                density_value =  density_EoS[i-1]*np.exp(
                    np.log(pressure/pressure_EoS[i-1])*np.log(density_EoS[i-1]/
                                                              density_EoS[i])
                    /np.log(pressure_EoS[i-1]/pressure_EoS[i]))
                return density_value
            else: 
                #If pressure higher than any value in our EoS -> No extrapolation
                return density_EoS[last_i]

def energy_density(pressure):
    """Returns energy density [km-2] as a function of pressure [km-2]
    Only interpolation, no extrapolation
    Same procedure as density(pressure)"""
    if pressure < 0:
        return 0 #No negative pressures
    else:
        #Firs index where the pressure of the EoS is higher than actual pressure
        i = np.argmax(pressure_EoS > pressure)
        if i == 0: #If is is the first index we dont extrapolate to lower values
            return energydens_EoS[0]
        else:
            last_i = len(energydens_EoS) - 1
            if i <= last_i: #Interpolate if 0 < i <= last_index
                energy_density_value =  energydens_EoS[i-1]*np.exp(
                    np.log(pressure/pressure_EoS[i-1])*np.log(energydens_EoS[i-1]/
                                                              energydens_EoS[i])
                    /np.log(pressure_EoS[i-1]/pressure_EoS[i]))
                return energy_density_value
            else: 
                #If pressure higher than any value in our EoS -> No extrapolation
                return energydens_EoS[last_i]
            
def pressure(density):
    """Returns pressure [km-2] as a function of density [fm-3]
    Only interpolation, no extrapolation
    Same procedure as density(pressure)"""
    if density < 0:
        return 0 #No negative densities
    else:
        #Firs index where the density of the EoS is higher than actual density
        i = np.argmax(density_EoS > density)
        if i == 0: #If is is the first index we dont extrapolate to lower values
            return pressure_EoS[0]
        else:
            last_i = len(pressure_EoS) - 1
            if i <= last_i: #Interpolate if 0 < i <= last_index
                pressure_value =  pressure_EoS[i-1]*np.exp(
                    np.log(density/density_EoS[i-1])*np.log(pressure_EoS[i-1]/
                                                              pressure_EoS[i])
                    /np.log(density_EoS[i-1]/density_EoS[i]))
                return pressure_value
            else: 
                #If density higher than any value in our EoS -> No extrapolation
                return pressure_EoS[last_i]

def TOV_equations(dependent_variables, independent_variable):
    """Input: List of dependent variables [M, PHI, P, A] and indep_var -> R
    Output: Dervitatives equations array"""
    r = independent_variable
    M = dependent_variables[0]
    #PHI = dependent_variables[1] #Not needed
    P = dependent_variables[2]
    #A = dependent_variables[3] #Not needed
    
    #Density and energy density as function of pressure
    n = density(P)
    eps = energy_density(P)
    #Derivative equations
    dmdx = 4.0 * np.pi * r**2 * eps
    dphidx = (M + 4.0 * np.pi * r**3 * P) / (r**2 * (1.0 - 2.0 * M / r))
    dpdx = -(eps + P) * dphidx
    dadx = (4.0 * np.pi * r**2 * n) / np.sqrt(1.0 - 2.0 * M / r)
    derivatives_array = np.array([dmdx, dphidx, dpdx, dadx])
    return derivatives_array

def RK4_1step(dep_var, indep_var, step, deriv_equations):
    """4th order Runge Kutta
    Note: deriv_equations expect the function TOV_equations"""
    n = len(dep_var)
    #Creating method parameters
    new_dep_var = np.zeros(n);
    k1 = np.zeros(n); k2 = np.zeros(n); k3 = np.zeros(n); k4 = np.zeros(n)
    
    k1 = deriv_equations(dep_var, indep_var)
    k2 = deriv_equations(dep_var + 0.5 * step * k1, indep_var + 0.5 * step)
    k3 = deriv_equations(dep_var + 0.5 * step * k2, indep_var + 0.5 * step)
    k4 = deriv_equations(dep_var + step * k3, indep_var + step)
    
    new_dep_var = dep_var + (step / 6.0) * (k1 + 2.0*k2 + 2.0*k3 +k4)
    
    return new_dep_var

def TOV_init(initial_radius, initial_density):
    """Returns the initial vector for the dependent variables"""
    dep_var = np.zeros(4)
    #Small contribution for the mass because we are not at exactly R = 0
    dep_var[0] = (4.*np.pi/3.)*energy_density(pressure(initial_density))*initial_radius**3
    #We can choose phi(r=0)=0 and add a term correction because we are not at r=0
    dep_var[1] = (2.*np.pi/3.)*(energy_density(pressure(initial_density))+
                                3.*pressure(initial_density))*initial_radius**2
    #Same for pressure
    dep_var[2] = (pressure(initial_density)-(2.0 * np.pi / 3.0) * 
                  (energy_density(pressure(initial_density)) + 
                   pressure(initial_density)) * 
                  (energy_density(pressure(initial_density)) + 3.0 * 
                   pressure(initial_density)) * initial_radius**2)
    #Same for number of baryons
    dep_var[3] = (4.0 * np.pi / 3.0) * initial_density * initial_radius**3                                 
    
    return dep_var

def TOV_integrate(dep_var0, r0, step, deriv_equations):
    """Integrate the TOV equations, returns (R, M, Phi, P, Ab)"""
    loop_limit = 100000
    
    n = len(dep_var0)
    output = np.zeros(n+1) #Variable extra for radius
    variables = dep_var0 #Initialize the variables
    r = r0
    dr = step
    #Store solutions
    sol = np.zeros((loop_limit, n))
    sol[0] = variables
    
    counter_loop = 0
    while sol[counter_loop, 2]>pressure_EoS[0] and counter_loop<loop_limit-1 :
        #Loop stops when we reach the lowest pressure of the EoS or reach the limit
        variables = RK4_1step(variables, r, dr, deriv_equations)
        counter_loop += 1
        sol[counter_loop] = variables
        r += dr
    if sol[counter_loop, 2]<0:#NOt negative solution, take the anterior index
        counter_loop -= 1
    output[0] = r
    output[1 : n+1] = sol[counter_loop]
    return output
"""----------------------------------------------------------------------------
                                    Calculation 1
----------------------------------------------------------------------------"""

lowdens_EoS = read_EoS('bps.dat')
highdens_EoS = read_EoS('beta_eos.dat')

#Assert both equations are in the same units
combined_EoS = join_EoS(lowdens_EoS, highdens_EoS)

#Taking each magnitude with the right units
density_EoS    = combined_EoS[:,0]                  #Units: fm^{-3}
pressure_EoS   = combined_EoS[:,1]*MeVfm_3Tokm_2    #Units: km^{-2}
energydens_EoS = combined_EoS[:,2]*MeVfm_3Tokm_2    #Units: km^{-2}

central_density_min = 0.12
central_density_max = 1.2
central_density = central_density_min

initial_r = 0.00001
dr = 0.001 #step size for the RK4

radii = []
masses = []
iloop = 0
print('\nIniatializing TOV Calculation\n\nLoopNumber; CentralPressure; Radius; Mass\n')

while central_density<central_density_max:
    #Get initial variables
    dependient_var = TOV_init(initial_r, central_density)
    #Solve
    solution = TOV_integrate(dependient_var, initial_r, dr, TOV_equations)
    
    radii.append(solution[0])
    masses.append(solution[1]*kmToMsun)
    print(iloop, central_density, solution[0], solution[1]*kmToMsun)
    #1% increment of the central density
    central_density += 0.01*central_density
    iloop += 1

radii = np.array(radii)
masses = np.array(masses)
#Determine max mass and its radius
max_mass = np.max(masses)
radius_maxmass = radii[np.argmax(masses)]
print('\nMaximum Mass: %.4f SolarMasses\nRadius %.4f km' %(max_mass,radius_maxmass))

plt.figure(figsize=(8,6))
plt.plot(radii, masses, 'k-', markersize = 2)
plt.plot(radius_maxmass, max_mass, 'ro')
plt.xlabel('Radius [km]', fontsize=15)
plt.ylabel(r"Mass [$M_{\odot}$]", fontsize=15)


"""----------------------------------------------------------------------------
                                    Calculation 2
----------------------------------------------------------------------------"""

lowdens_EoS = read_EoS('bps.dat')
highdens_EoS = read_EoS('high_eos.dat')
#From dyne/cm2 to MeV/fm3
dinacm2tomevfm3 = 1e-33/1.602
highdens_EoS[:,1] = highdens_EoS[:,1]*dinacm2tomevfm3
#From g/cm3 to MeV/fm3
gcm3tomevfm3 = 1e-12/1.783
highdens_EoS[:,2] = highdens_EoS[:,2]*gcm3tomevfm3


#Assert both equations are in the same units
combined_EoS = join_EoS(lowdens_EoS, highdens_EoS)

#Taking each magnitude with the right units
density_EoS    = combined_EoS[:,0]                  #Units: fm^{-3}
pressure_EoS   = combined_EoS[:,1]*MeVfm_3Tokm_2    #Units: km^{-2}
energydens_EoS = combined_EoS[:,2]*MeVfm_3Tokm_2    #Units: km^{-2}

central_density_min = 0.12
central_density_max = 1.35
central_density = central_density_min

initial_r = 0.00001
dr = 0.001 #step size for the RK4

radii = []
masses = []
iloop = 0
print('\nIniatializing TOV Calculation\n\nLoopNumber; CentralPressure; Radius; Mass\n')

while central_density<central_density_max:
    #Get initial variables
    dependient_var = TOV_init(initial_r, central_density)
    #Solve
    solution = TOV_integrate(dependient_var, initial_r, dr, TOV_equations)
    
    radii.append(solution[0])
    masses.append(solution[1]*kmToMsun)
    print(iloop, central_density, solution[0], solution[1]*kmToMsun)
    #1% increment of the central density
    central_density += 0.01*central_density
    iloop += 1

radii = np.array(radii)
masses = np.array(masses)
#Determine max mass and its radius
max_mass = np.max(masses)
radius_maxmass = radii[np.argmax(masses)]
print('\nMaximum Mass: %.4f SolarMasses\nRadius %.4f km' %(max_mass,radius_maxmass))


plt.plot(radii, masses, 'b-', markersize = 2)
plt.plot(radius_maxmass, max_mass, 'ro')

plt.savefig('mass_radius_both.png')
plt.show()