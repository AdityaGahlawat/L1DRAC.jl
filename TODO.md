## TODO List (Aditya & Sambhu)

Check the todos at the bottom of README.md

- [x] Create functions to define all the constants used in the manuscript  
- [x] Create a function to define the $\rho_r$ and $\rho_a$ condition checks for a given set of assumption parameters  
- [x] Test with a random set of values for the assumption constants and verify that all the $\Delta$'s return valid values (no NaN or absurd numbers)  
- [x] Verify that the constants match those in the manuscript  
- [x] Identify the correct assumption constants for the vector fields defined for the Double Integrator example  
- [x] Find $\rho_r$ and $\rho_a$ for the above vector fields using binary search  
- [x] Create a function to define the $\omega$ condition check for a given set of assumption parameters  
- [ ] Formulate a nonlinear optimization problem to find the minimum $\rho_r$, $\rho_a$, and $\omega$ that satisfy all the conditions
- [ ] Using the bounds, set up the computation of control parameters
- [ ]  Test on Covariance steering example  
- [ ] Plot the sample paths of the nominal and true system along with the covariance ellipsoids (using a package)  
- [ ] Compute the $2$-Wasserstein distance and check whether the theoretical bound is violated for any time $t \leq T$  


