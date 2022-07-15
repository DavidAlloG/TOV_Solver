# TOV_Solver
Solves the equations of Tolman-Oppenheimer-Volkov for Neutron Stars taking into account the simetry energy.

Small script to compare the neutron stars maximum mass if we take into account symmetric or asymmetric nuclear matter, that I use to study Symmetry Energy (pdf file).

The script takes the 3 different equations of state to calculate the realtions between density, energy density and pressure in the interior of neutron stars.
* bps.dat: Data for low density matter equation of state.
* beta_eos.dat: Data for high density symmetric matter equation of state.
* high_eos.dat: Data for high density asymmetric matter equation of state.

With the equation of state we are capable of solve the TOV equations using numerical methods (RK4).

$$\frac{dP}{dr} = -\frac{GM\epsilon}{r^2}\frac{(1+P/\epsilon)(1+4\pi r^3 P/M)}{1-2GM/r}$$
$$\frac{dM(r)}{dr} = 4\pi\epsilon r^2$$

We can obtain the following maximum mass - radius restrictions in each model and compare them:
![mass_radius_both](https://user-images.githubusercontent.com/109382404/179297530-2aeba554-1194-4c99-a886-23d6105e1b05.png)
