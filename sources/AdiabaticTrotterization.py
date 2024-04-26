# -*- coding: utf-8 -*-
"""
@author: QUTAC Quantum

SPDX-FileCopyrightText: 2024 QUTAC

SPDX-License-Identifier: Apache-2.0

"""
import numpy as np
from pyqubo import Binary
from qiskit import QuantumCircuit, Aer, transpile
from qiskit import QuantumCircuit
import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

class TrotterizedADB:
    """
    A class for implementing the trotterized adiabtic evolution
    
    """

    def __init__(self, article_weight, knapsack_capacity, article_value, optimal, 
                 n_layers, opt_method, qubo_formulation, convl, shots_scale):
        """
        Constructs all the necessary attributes for the TrotterizedADB object.

        Parameters
        ----------
            article_weight : numpy array
                contains the weights for all items
            knapsack_capacity : numpy array
                contains the capacity of each knapsack
            article_value : numpy array
                contains the value (reward) for filling each item in a knapsack
            optimal : int/float
                optimal objective value
            n_layers : int
                number of QAOA layers
            opt_method : str
                method for optimization: use 'expval', 'cVaR', or 'min'
            qubo_formulation: str
                type of  qubo ('standard_qubo' 'no_slack_qubo')
            convl : int
                confidence level if cVaR is used as optimization method. 
            shots_scale: str
                scaling factor for computing the number of shots for sampling, 'linear', 'quadratic', 'constant'

        Returns
        ----------
            None

        """

        self.article_weight = article_weight
        self.article_value = article_value
        self.knapsack_capacity = knapsack_capacity
        self.n_layers = n_layers
        
        self.opt_method = opt_method
        self.optimal=optimal
        self.full_state = False
        self.qubo_formulation = qubo_formulation
        self.convl = convl
        self.shots_scale = shots_scale


        # factors of 2 added to make sure that Ising coefficients are integers (off-diagonal QUBO terms need a factor of 4, but get another factor of 2 from multiplying terms in the sum). 
        # Note that this is not relevant when the QUBO is normalized
        self.single_penalty = (np.sum(article_value)+np.sum(article_weight))*2*50
        self.capacity_penalty = (np.sum(article_value)+np.sum(article_weight))*2 
        self.objective_weight = 2

        self.number_of_knapsacks = len(self.knapsack_capacity)
        self.number_of_articles  = len(self.article_weight)

        (self.variables,
         self.quadratic,
         self.var_to_index,
         self.index_to_var,
         self.model,
         self.offset,
         self.Q) = self._create_qubo()
        self.num_qubits = len(self.variables)

        self.h, self.J, self.final_offset = self._qubo_to_ising()
        
        self.max_coeff = max(
            max(list(np.abs(list(self.h.values())))),
            max(list(np.abs(list(self.J.values()))))
            )


        self.shots=0
        if self.shots_scale =='linear':
            self.shots = int(500*self.num_qubits)
        elif self.shots_scale =='quadratic':
            self.shots = int(500*self.num_qubits**2)
        elif self.shots_scale =='constant':
            self.shots = 10000
        
        self.minimum_value = 10**9
        self.minimum_sol = 10**9
        self.optimized_sampled_states = []

        global Nfeval

    ##########################################################################################
    # create the QUBO model for knapsack problem
    ##########################################################################################
    def _create_qubo(self):
        """
        Internal method to create the QUBO model for multi-knapsack problem

        Parameters
        ----------
            None
        Returns
        ----------
            variables: list
                list of QUBO variables names
            quadratic:  dict
                dictionary mapping a combination of binary variables to their coefficient
            var_to_index:  dict
                dictionary mapping the QUBO variables to the index number in QUBO matrix
            index_to_var:  dict
                dictionary mapping the index number in QUBO matrix to QUBO variables
            model: <pyqubo object>
                pyqbo QUBO model object
            offset: int/float
                constant offset in the QUBO
            Q: numpy array
                the QUBO matrix
        """
        x = dict()
        for i in range(self.number_of_knapsacks):
            for j in range(self.number_of_articles):
                x[(i, j)] = Binary(f'x_{i}_{j}')

        self.H_single = 0
        for j in range(self.number_of_articles):
            temp = sum([x[i, j] for i in range(self.number_of_knapsacks)])
            self.H_single += temp * (temp - 1)
        self.H_single *= self.single_penalty

        self.H_obj = sum([-self.article_value[i, j] * x[i, j] for i in range(self.number_of_knapsacks) for j in
                          range(self.number_of_articles)])

                 
        if self.qubo_formulation == 'standard_qubo':
            y = dict()
            for i in range(self.number_of_knapsacks):
                num_slack_bits = len(bin(self.knapsack_capacity[i])) - 2
                for b in range(num_slack_bits):
                    y[i, b] = Binary(f'y_{i}_{b}')
            self.H_capacity = 0
            for i in range(self.number_of_knapsacks):
                temp = sum([self.article_weight[j] * x[i, j] for j in range(self.number_of_articles)])
                num_slack_bits = len(bin(self.knapsack_capacity[i])) - 2
                temp += sum([2 ** b * y[i, b] for b in range(num_slack_bits)])
                temp -= self.knapsack_capacity[i]
                self.H_capacity += temp ** 2
        elif self.qubo_formulation=='no_slack_qubo':   
            self.H_capacity = 0
            for i in range(self.number_of_knapsacks):
                temp = sum([self.article_weight[j] * x[i, j] for j in range(self.number_of_articles)])
                temp -= self.knapsack_capacity[i]
                self.H_capacity += temp ** 2
        self.H_capacity *= self.capacity_penalty # 5=B
    
        
        self.H_obj *= self.objective_weight
        H = self.H_single + self.H_capacity + self.H_obj

        model = H.compile()
        terms = model.to_qubo()#ising()
        quadratic = terms[0]
        offset = terms[1]
        variables = model.variables
        Q = np.zeros((len(variables),len(variables)))

        ##################################################################
        var_to_index = dict([(n, i) for i, n in enumerate(variables)])
        index_to_var = dict([(i, n) for i, n in enumerate(variables)])

        for (k,l) in quadratic:
            i=var_to_index[k]
            j=var_to_index[l]
            Q[i,j] = quadratic[(k,l)]
        Q = (Q+Q.T)/2

        return variables, quadratic, var_to_index, index_to_var, model, offset, Q
    
    ########################################################################################
    # Transform QUBO to Ising model
    ########################################################################################
    def _qubo_to_ising(self):
        """
        Internal method to convert the QUBO model to an ISING model

        Parameters
        ----------
            None

        Returns
        ----------
            h : list
                Linear coefficients
            J : dict
                Quadratic coefficients
            final_offset : float
                Offset from the formulation

        """
        h = dict()
        for idxi in range(self.num_qubits):
            h[idxi] = 0.0
        J = dict()
        linear_offset = 0.0
        quadratic_offset = 0.0
        for idxi in range(self.num_qubits):
            h[idxi] -= 0.5*self.Q[idxi, idxi]
            linear_offset += self.Q[idxi, idxi]
            for indxj in range(idxi+1,self.num_qubits):
                J[(idxi, indxj)] = 0.5*self.Q[idxi, indxj]
                h[idxi] -= 0.5*self.Q[idxi, indxj]
                h[indxj] -= 0.5*self.Q[idxi, indxj]
                quadratic_offset += 2*self.Q[idxi, indxj]

        final_offset = 0.5*linear_offset + 0.25*quadratic_offset + self.offset

        return h, J, final_offset
    

    ##########################################################################################
    # Unitary for problem hamiltonian
    ##########################################################################################
    def U_C(self, qc, gamma):
        """
        Method to create `U_C` in TAE (this unitary operator is similar to the phase-separator in QAOA)

        Parameters
        ----------
            qc : qiskit object
                the quantum circuit for QAOA
            gamma : float
                the angle for the unitary operator U_C

        Returns
        ----------
            None

        """
        for qubit_num in self.h:
            coefficient = self.h[qubit_num]/self.max_coeff
            qc.rz(2 * coefficient * gamma, qubit_num)
        for (qubit1, qubit2) in self.J:
            coefficient = self.J[(qubit1, qubit2)]/self.max_coeff
            qc.cx(qubit1, qubit2)
            qc.rz(2*coefficient * gamma, qubit2)
            qc.cx(qubit1, qubit2)

    ##########################################################################################
    # Unitary for mixing hamiltonian
    ##########################################################################################
    def U_B(self, qc, beta):
        """
        Method to create `U_B` for TAE (this unitary operator is similar to the mixer unitary operator in QAOA) 

        Parameters
        ----------
            qc : qiskit object
                the quantum circuit for QAOA
            beta : float
                the angle for the unitary operator U_B

        Returns
        ----------
            None

        """
        for qubit in range(self.num_qubits):
            qc.rx(2 * beta, qubit)

    ##########################################################################################
    # Sample from circuit
    ##########################################################################################
    def sample_circuit(self, gammas, betas):
        """
        Method to create the complete TAE quantum circuit and sample the quantum state for parameterized rotation angles

        Parameters
        ----------
            gammas : parameterized rotation angle for problem unitary operator
            betas : parameterized rotation angle for mixer Hamiltonian unitary operator

            
        Returns
        ----------
             sampled_states : dictionary containing all the sampled basis states along with the counts for each state
        """
        qc = QuantumCircuit(self.num_qubits)

        for qubit in range(self.num_qubits):
            qc.x(qubit)
            qc.h(qubit)
        # p instances of unitary operators
        for i in range(self.n_layers):            
            self.U_C(qc, gammas[i])
            self.U_B(qc, betas[i])

        qc=qc.reverse_bits()
        
        if self.full_state:
            backend = Aer.get_backend("statevector_simulator")
            sampled_states = backend.run(qc).result().get_statevector(qc)
        else:
            qc.measure_all()
            simulator = Aer.get_backend('aer_simulator')
            qc = transpile(qc, simulator)
            result = simulator.run(qc, shots=self.shots).result()
            sampled_states = result.get_counts(qc)      

        return sampled_states
    
  
    ##########################################################################################
    # Compute cost and penalty for final evaluation
    ##########################################################################################
    def final_energy_penalty(self, x_val, qubo):
        """
        Method to compute the final QUBO penalty value and QUBO cost after optimization

        Parameters
        ----------
            x_val : string
                bitsring containg the solution bits (0/1)
            qubo: str
                the qubo_formulation type `no_slack_qubo` or `standard_qubo`

        Returns
        ----------
             cost : float
                total QUBO energy including objective and penalty terms
             penalty : float
                qubo penalty value
        """
        x = dict()
        y = dict()
        for l in range(len(x_val)):
            variable = self.index_to_var[l]
            (i,j)=int(variable.split('_')[1]), int(variable.split('_')[2])
            if 'x' in variable:
                x[(i,j)]=int(x_val[l])
            elif 'y' in variable:
                y[(i,j)]=int(x_val[l])
        
        energy_single = 0
        for j in range(self.number_of_articles):
            temp = sum([x[i, j] for i in range(self.number_of_knapsacks)])
            energy_single += temp * (temp - 1)
        energy_single *= self.single_penalty/2

        energy_capacity = 0
        if qubo=='no_slack_qubo':
            energy_capacity = 0
            for i in range(self.number_of_knapsacks):
                temp = sum([self.article_weight[j] * x[i, j] for j in range(self.number_of_articles)])
                temp += np.max([0, self.knapsack_capacity[i]-temp])
                temp -= self.knapsack_capacity[i]
                energy_capacity += temp ** 2
        elif qubo=='standard_qubo': 
            energy_capacity = 0
            for i in range(self.number_of_knapsacks):
                temp = sum([self.article_weight[j] * x[i, j] for j in range(self.number_of_articles)])
                num_slack_bits = len(bin(self.knapsack_capacity[i])) - 2
                temp += sum([2 ** b * y[i, b] for b in range(num_slack_bits)])
                temp -= self.knapsack_capacity[i]
                energy_capacity += temp ** 2
        
        
        energy_capacity *= self.capacity_penalty/2

        objective = sum([-self.article_value[i, j] * x[i, j] for i in range(self.number_of_knapsacks) for j in
                          range(self.number_of_articles)])
        
        objective *= self.objective_weight/2 
        
        penalty = energy_single+energy_capacity
        cost = penalty+objective
        return cost, penalty
    
    ##########################################################################################
    # Compute the objective function for optimization i.e., expectation value 
    ##########################################################################################
    def objective_function(self, sampled_states, qubo):
        """
        Method to compute the expectation value of the sampled quantum states

        Parameters
        ----------
            sampled_states : dict
                a dictionary mapping all the measured basis-states to their probability
            qubo: str
                the qubo_formulation type `no_slack_qubo` or `standard_qubo`
            
        Returns
        ----------
            objective_value: float
                the objective value depending on `self.opt_method`
            min_sol: string
                the bitsring containing the solution with minimum energy
            min_val: float
                the minimum objective value among all `samples_states`
        """

        all_cost_values = []
        all_bitstrings=[]
        costs = []
        if self.full_state:
            for i in range(len(sampled_states)):
                probability = np.abs(sampled_states[i])**2
                solution_string = bin(i).replace('0b', '').zfill(self.num_qubits)
                x_val = list([int(bits) for bits in solution_string])
                cost, _ = self.final_energy_penalty(x_val, qubo)
                all_cost_values.append(cost*probability)
                all_bitstrings.append(x_val)
                costs.append(cost)                
        else:
            for state in sampled_states:
                probability = sampled_states[state]/self.shots
                x_val = list([int(bits) for bits in state])
                cost, _ = self.final_energy_penalty(x_val, qubo)
                all_cost_values.append(cost*probability)
                all_bitstrings.append(x_val)
                costs.append(cost)

        min_sol = all_bitstrings[np.argmin(costs)]
        min_val = min(costs)

        if self.opt_method == 'expval':
            objective_value = np.sum(all_cost_values)
        elif self.opt_method == 'cVaR':
            sorted_cost_values = [x for _, x in sorted(zip(costs, all_cost_values))]
            max_ind=int(self.convl*len(sampled_states)) #number of shots to consider for evaluation
            objective_value = np.sum(sorted_cost_values[:max_ind])
        elif self.opt_method == 'min':
            objective_value=min_val
            

        return objective_value, min_sol, min_val

    ##################################################################################
    # Evaluate final results
    ##################################################################################
    def compute_final_metrics(self, sampled_states, qubo):
        """
        Method to compute the final metrics from the optimized `sampled_states`

        Parameters
        ----------
            sampled_states : dict
                a dictionary mapping all the measured basis-states to their probability
            qubo: str
                the qubo_formulation type `no_slack_qubo` or `standard_qubo`

        Returns
        ----------
            prob_optimal: float
                the sum of probabilities of all the optimal states sampled in `sampled_states`
            num_optimal: int
                the number of times an optimal state is sampled in `sampled_states`
            prob_optimal_90: float
                the sum of probabilities of all the valid states in `sampled_states` whose energy is equal to or better than 90% of the optimal energy
            num_optimal_90: int
                the number of times, a valid state whose energy is equal to or better than the 90% of optimal energy is sampled in `sampled_states`
            num_valid: int
                the number of times, a valid state is sampled in `sampled_states`
        """
        num_optimal = 0
        num_valid=0
        prob_optimal = 0 
        num_optimal_90 = 0
        prob_optimal_90 = 0
        for state in sampled_states:
            x_val = list([int(bits) for bits in state])
            (cost, penalty) = self.final_energy_penalty(x_val, qubo)
            probability=sampled_states[state]/self.shots
            num_sols=sampled_states[state]
            if  penalty <= 10**-4:
                num_valid += num_sols        
                if (-1*cost-penalty) >= 0.99999999999*self.optimal:
                    num_optimal += num_sols
                    prob_optimal += probability
                if (-1*cost-penalty) >= 0.9*self.optimal:
                    num_optimal_90 += num_sols
                    prob_optimal_90 += probability
        return prob_optimal, num_optimal, prob_optimal_90, num_optimal_90, num_valid
    
    
