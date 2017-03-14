#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 22:21:28 2017

@author: Daniel Speckhard, DTS@stanford.edu

This library was created at the Stanford Sustainable Systems Lab. This library
includes functions to estimate the topology of an electrical grid given voltage 
measurements at nodes.
"""
import numpy as np
import networkx as nx
from sklearn.metrics import mutual_info_score
from est_MI import est_MI_JVHW
class grid_est:
    
    def __init__(self, true_branches, vmag_matrix):
        """ Inputs:
        --- true_branches: This is matrix storing the node lables of each 
        pair of nodes connected by a true branch in the true tree 
        structure. Ex. [(2,1), (3,1)] would indicate that nodes 2 and 3 are 
        both connected to node 1. The size of the matrix should be 
        ((number of nodes - 1) x 2).
        --- vmag_matrix: This matrix should contain the voltage magnitude 
        data at different timepoints and different nodes. The matrix should
        be size ((number of nodes) x (number of data points at each node)). 
        Ouputs:
        --- num_buses
        --- MI_matrix: This matrix stores mutual information values between 
        nodes. The matrix is size (number of nodes x number of nodes). Since 
        mutual information is symmetric (MI(3,4) = MI(4,3)), the lower 
        triangular values are filled-in and the upper-triangular values are 
        left at zero. The diagonal values are not useful as part of estimation 
        and also left as zero.
        --- est_branches: These are the branches that are estimated by the 
        algorithm. The size of this matrix is ((number of nodes - 1) x 2). 
        """
        self.true_branches = true_branches -1
        self.vmag_matrix = vmag_matrix
        self.MI_matrix = 'undefined' # Leave this undefined until later.
        self.num_buses = len(self.vmag_matrix[0,:])
        self.est_branches = np.zeros((self.num_buses - 1, 2))
        self.graph = 'undefined'
        
    def find_gaussian_entropy(self):
        """ This function finds the entropy by approximating the voltage 
        magnitude data at different nodes as a Gaussian distribution.
        Inputs:
        --- self.vmag_data: This is the voltage magnitude data at different 
        nodes and timepoints. 
        Outputs:
        --- entropy_vec: This is a vector size (1 x number of buses) with 
        values of the vector, v(i) equal to the entropy at node(i). The 
        entropy is caluclated by approximating the data at a node in the
        input data matrix, node_volt_matrix(:,i), as a Gaussian distribution.
        """
        entropy_vec = np.zeros((self.num_buses))
        for i in range(0,self.num_buses):
           # We use the equation H(X) = k/2*(1+ln(2*pi)) + 1/2*ln|Sigma|. 
           # Where k is the dimension of the vector X and Sigma is the 
           # covariance matrix. Note, here we only have one dimension, 
           # vmag(i), in our entropy calc. This is not the case if we would 
           # like to include phase information into our calculation.
           #print(i)
           entropy_vec[i] = 0.5*(1+np.log(2*np.pi))+0.5*np.log(np.var(self.vmag_matrix[:,i]))
           #print(entropy_vec[i])
                       
        return entropy_vec
                       
    def find_joint_gaussian_entropy(self):
        """ This function calculates the pair-wise mutual information between 
        two nodes in the input data matrix, node_volt_matrix. The mutual 
        information is claculated by appoximating the distribution of data at 
        each node as a Gaussian distribution. 
        
        Input:
        ----- node_volt_matrix: A matrix containing the voltage magnitude data.
        The number of columns should be the number of nodes in the system. The
        number of rows in the matrix corresponds to the number of datapoints
        where the voltage magnitude is measured across all nodes.
        Output:
        % Ouput:
        ----- joint_entropy_matrix: This matrix contains the joint entropy
        between two nodes from the input matrix, node_volt_matrix. The 
        matrix is a lower triangular matrix since the joint entropy is
        symmetric, i.e., Entropy(i,j) = Entropy(j,i). The joint entropy of the
        same node, Entropy(i,j) is not used in the chow-liu algorithm and
        therefore these values (the diagonal of joint_entropy_matrix) are set 
        to zero. The entropy is caluclated by approximating the data at a node 
        in the input data matrix, node_volt_matrix(:,i), as a Gaussian 
        distribution. The size of the matrix is (number of nodes x number of 
        nodes). We use the equation joint_entropy([X,Y]) = k/2*(1+ln(2*pi)) + 
        1/2*ln|Sigma|. Where k is the dimension of the vector [X,Y] and Sigma 
        is the covariance matrix.
        """
        # Initialize the joint entropy matrix, note the size it determined
        # by the number of buses in the grid.
        joint_entropy_matrix = np.zeros((self.num_buses,self.num_buses))
        
        # We avoid calculating the joint_entropy values using the same node 
        # twice. Therefore the diagonal values for the join_entropy_matrix 
        # are not calculated.
        for i in range(1,self.num_buses):
            for k in range(0,(i)):
        
        # We use the equation joint_entropy([X,Y]) = k/2*(1+ln(2*pi)) +
        # 1/2*ln|Sigma|. Where k is the dimension of the vector [X,Y] and 
        # |Sigma| is the deteriminant of the covariance matrix. For two 
        # nodes, where each node contributes it's voltage magntiude data, 
        # k, the dimension of [X,Y] is equal to two.
        
                det_of_cov_matrix = np.linalg.det(np.cov(
                        self.vmag_matrix[:,i],self.vmag_matrix[:,k]));
        
        # Let's check if the determinant value is very close to zero
        # and negative. If this is true, a numerical rounding error has 
        # likley happened and we want to avoid getting a log of a negative
        # number.
                if (det_of_cov_matrix <= 0) and (
                   det_of_cov_matrix >  -0.0001):
            # Then we assume this is a numerical error and we make the
            # mutual information negative by setting the joint entropy
            # value to a very negative number. This is abritrary and this
            # case should never occur in practice unless we end up
            # computing the joint entropy of two nodes that are labelled
            # differently but contain the same data.
                    joint_entropy_matrix[i,k] = -1E3;
                else: # Othewrwise we compute as normal.
                    joint_entropy_matrix[i,k] = 2/2*(
                                        1+np.log(2*np.pi))+0.5*np.log(
                                        det_of_cov_matrix);
        return joint_entropy_matrix
    
    def find_MI_mat(self, entropy_vec, joint_entropy_matrix):
        """ The function takes in as input the single node entropy values 
        H(j) and joint entropy matrix values, H(j,k). The output is a lower 
        triangular matrix that contains the mutual information MI(j,k). Since
        mutual information is symmetric we only have to calculate the lower
        triangular values. 
        Inputs:
        ---- single_node_entropy_vec: This is a vector size (1 x number of
        buses) with values of the vector, v(i) equal to the entropy at 
        node(i). The entropy is caluclated by approximating the data at a 
        node in the input data matrix, node_volt_matrix(:,i), as a Gaussian 
        distribution.   
        ---- joint_entropy_matrix: This matrix contains the joint entropy
        between two nodes. The matrix is a lower triangular matrix since the
        joint entropy is symmetric, i.e., Entropy(i,j) = Entropy(j,i). 
        The joint entropy of the same node, Entropy(i,j) is not used in the 
        chow-liu algorithm and therefore these values (the diagonal of 
        joint_entropy_matrix) are set to zero. % The entropy is caluclated by 
        approximating the data at a node in the input data matrix, 
        node_volt_matrix(:,i), as a Gaussian distribution. The size of the 
        matrix is (number of nodes x number of nodes). We use the
        equation joint_entropy([X,Y]) = k/2*(1+ln(2*pi)) + 1/2*ln|Sigma|. Where k
        is the dimension of the vector [X,Y] and Sigma is the covariance 
        matrix.
        Outputs:
        ---- MI_matrix: This matrix is size (number of nodes x number of 
        nodes). The matrix is lower triangular, meaning the diagonal and 
        elements above the diagonal are zero since the mutual information 
        between two nodes MI(i,j) = MI(j,i) is symmetric and the 
        self-information MI(i,i) is not used in the algorithm since we don't 
        connect a node to itself (to avoid cycles). The mutual information can
        be caluclated from the joint entropy and the single node entropy by 
        recalling MI(i,j) = entropy(i) +  entropy(j) - joint_entropy(i,j). """
        # Note we only populate the lower triangular values of the mutual
        # information matrix since we don't care about self-information values 
        # and the mutual information is symmetric.
        self.MI_matrix = np.zeros((self.num_buses,self.num_buses))
        for i in range(1,self.num_buses):
            for k in range(0,(i)):
        # The mutual information can be caluclated from the joint entropy
        # and the single node entropy by recalling MI(i,j) = entropy(i) +
        # entropy(j) - joint_entropy(i,j).
                   self.MI_matrix[i,k] = entropy_vec[i] + entropy_vec[k] - \
                              joint_entropy_matrix[i,k]
                          
    def find_discrete_MI(self, bins):
        """ Input:
            ---- self.vmag_matrix: This is a matrix size number of data-points
            x number of nodes. The matrix contains the voltage magntiude or the
            change in voltage magntiude (if take_deriv() has been called). 
            ---- bins: This controls how many bins to bin the continuous data
            from self.vmag_matrix. We must discretize the data before taking 
            the discrete mutual information. This parameter will control
            the computation time of this function as well as the precision.
            Output:
            ----- self.MI_matrix: This matrix will be size number of nodes 
            x number of nodes. The matrix will be non-zero only for lower 
            triangular values since MI is symmetric and self-information values
            are of no use to us. The matrix will be calculated from the formula
            Sum_ij p(i,j)log(p(i,j)/(p(i)*p(j))).
        
            """
        def calc_discrete_MI(x, y, bins):
            c_xy = np.histogram2d(x, y, bins)[0]
            mi = mutual_info_score(None, None, contingency=c_xy)
            return mi
        self.MI_matrix = np.zeros((self.num_buses,self.num_buses))
        for i in range(1,self.num_buses):
            for j in range(0,(i)):
                   self.MI_matrix[i,j] = calc_discrete_MI(
                           self.vmag_matrix[:,i],self.vmag_matrix[:,j],bins)

    def find_JVHW_MI(self):
        
        """ Input:
            ---- self.vmag_matrix: This is a matrix size number of data-points
            x number of nodes. The matrix contains the voltage magntiude or the
            change in voltage magntiude (if take_deriv() has been called). 
            ---- bins: This controls how many bins to bin the continuous data
            from self.vmag_matrix. We must discretize the data before taking 
            the discrete mutual information. This parameter will control
            the computation time of this function as well as the precision.
            Output:
            ----- self.MI_matrix: This matrix will be size number of nodes 
            x number of nodes. The matrix will be non-zero only for lower 
            triangular values since MI is symmetric and self-information values
            are of no use to us. The matrix will be calculated using JVHW
            estimators for entropy.
            """
        self.MI_matrix = np.zeros((self.num_buses,self.num_buses))
        for i in range(1,self.num_buses):
            for j in range(0,(i)):
                self.MI_matrix[i,j] = est_MI_JVHW(
                self.vmag_matrix[:,i],self.vmag_matrix[:,j])
            
            
    def find_SDR(self):
        
        """ Input:
        ---- self. est_branches: This is a matrix size 
        (number of branches x 2) where each row is a pair of nodes 
        corresponding to an estimated branch from the chow-liu algorithm 
        output. The number of branches should be one less than the total 
        number of nodes.
        ---- self.true_branches: This is a matrix containing the true 
        branches in the system. The number of rows in this matrix corresponds 
        to the number of true branches in the system. THe number of columns 
        should be equal to two, which is the number of nodes involved in a 
        single branch.
        Output:
        ----- successful_branch_counter: This is a integer counter that tells 
        us how many branches have been correctly estimated by the algorithm.
        ----- SDR: This value is the successful detection rate (SDR) 
        percentage of the estimation algorithm. It equals the number of 
        correctly identified branches divided by the total number of branches.
        """
        
        successful_branch_counter =0.0
        #flipped_est_branches = np.fliplr(self.est_branches)
        for i in range(0,np.shape(self.est_branches)[0]):
            if ([self.est_branches[i,0],self.est_branches[i,1]]
                in self.true_branches.tolist()) or ([self.est_branches[i,1],
                                            self.est_branches[i,0]] 
                in self.true_branches.tolist()):
                
                successful_branch_counter += 1.
        
        sdr = np.divide(successful_branch_counter, len(self.est_branches[:,0]))
        
        return sdr, successful_branch_counter
    
    def take_deriv(self, deriv_step):
        """ This function finds the change in voltage magntiude for a variable
        difference of time-points. The change in voltage magnitude is 
        calculated as Delta_Vmag = Vmag(node x, time t) - Vmag(node x, time 
        t-deriv_step).
        Input:
        --- self.vmag_matrix: This is the voltage magnitude data at different 
        nodes. We want to transform this data so we end up with the change in 
        voltage magnitude from one time-point to the next for each node.
        --- deriv_step: This parameter decides how many time-points for which
        to compute the voltage magntiude difference at a node. Ex. if 
        deriv_step = 5, then the change in voltage magnitude will be 
        Delta_Vmag = Vmag(node x, time t) - Vmag(node x, time t-5)
        Ouput:
        --- self.vmag_matrix: This matrix now represents the change in voltage
        magnitude at a node from one time-point to the next. Each column 
        corresponds to a different node and each row to a time-point.
        """ 
        end = len(self.vmag_matrix[:,1])-1
        self.vmag_matrix = self.vmag_matrix[deriv_step+1:-1,:] - \
                                           self.vmag_matrix[1:end-deriv_step,:]
                                           
    def find_mst(self):
        """ This function finds the minimum spanning tree based on the 
        mutual information matrix. """
        
        # We create a fully connected graph with weights specified by the 
        # the mutual information values. 
        # However, we have to take the reciprical of the MI matrix values 
        # since nx only has min_span_tree as a fxn for python 2.7. We 
        # add a tiny number to each value so reciprocal is defined.
        MI_matrix = self.MI_matrix + 1e-12
        # We cretae a temporary nx. graph object that's a cyclical graph
        # with edge weights equal to MI values in the matrix.
        net = nx.from_numpy_matrix(np.reciprocal(MI_matrix), 
                                          create_using=nx.Graph())
        mst = nx.minimum_spanning_tree(net)
        self.est_branches = np.array(mst.edges())
        self.graph = mst
#        # Extract edges from mst
#        E = set(mst.edges())  # optimization
#        self.est_branches = [e for e in mst.edges() \
#                             if e in E or reversed(e) in E]
        
    def run_mst(self, MI_method):
        self.collapse_redundant_data()
        # Take deriv of data, with step-size = 1.
        self.take_deriv(1)
        # Find the etnropy
        if MI_method == 'gaussian':
            entropy_vec = self.find_gaussian_entropy()
            joint_entropy_matrix = self.find_joint_gaussian_entropy()
            self.find_MI_mat(entropy_vec, joint_entropy_matrix)
        if MI_method == 'discrete':
            bits = 8
            bins = 2**8
            self.discretize_signal(bits)
            self.find_discrete_MI(bins)
        if MI_method == 'JVHW':
            bits = 8
            bins = 2**8
            self.discretize_signal(bits)
            self.find_JVHW_MI()
        self.find_mst()
        
    def discretize_signal(self, bits):

        """ This function takes voltage magnitude data for the nodes in the system
    and returns the change in voltage magnitude between sucessive time-points
    for the nodes in the system. Since this process is similar to taking the
    deriative, the function is called consider_derivative.
    Input:
    ----- self.vmag_matrix: A matrix containing the voltage magnitude data.
    The number of columns should be the number of nodes in the system. The
    number of rows in the matrix corresponds to the number of datapoints
    where the voltage magnitude is measured across all nodes.
    ----- bits: Number of bins to which to discreteize data into. Meaning, if 
    bins = 10, we bin the data into 2^10 bins For this scheme we make each
    bin equal size. 
    Ouput:
    ----- self.vmag_matrix: This matrix is the size of the input
    node_volt_matrix. The values in this matrix are all integers. The values
    in are discretized, digitized_mat(i,j), are equal to the bin number
    which the value node_volt_matrix(i,j) falls into with respect to the
    binning process."""

        global_min = np.min(np.min(self.vmag_matrix));

        global_max = np.max(np.max(self.vmag_matrix));

        # Shift the data so new minimum is at zero.
        self.vmag_matrix = self.vmag_matrix - global_min;
        # Determine the bin-size for equally spaced bins.
        bin_size = np.divide((global_max-global_min),(2**bits - 1));
        # Now find new values for data:
        self.vmag_matrix = \
            np.round(np.divide(self.vmag_matrix,bin_size)).astype(int);
    

    def collapse_redundant_data(self):
        """ This function will check to see if there are any nodes that contain
        voltage measurements that are the same as another node. For instance, 
        if 40 nodes are given, this function will check if Node 1 has the same 
        data as Node 2. If Node Node 1 and Node 2 have the same data this 
        function will collapse the two nodes, meaning it will remove Node 2 
        and fix the list of true branches to replace any reference to Node 2 
        with Node 1.
        Inputs:
        ----- node_volt_matrix: A matrix containing the voltage magnitude data.
        The number of columns should be the number of nodes in the system. The
        number of rows in the matrix corresponds to the number of datapoints
        where the voltage magnitude is measured across all nodes.
        ----- true_branches: This is a matrix containing the true branches in
        the system. The number of rows in this matrix corresponds to the number
        of true branches in the system. THe number of columns should be equal 
        to two, which is the number of nodes involved in a single branch.
        Output: 
        ----- node_volt_matrix: This matrix is the node_volt_matrix dataset 
        with the redundant nodes removed. Therefore, the output matrix will be 
        different size than the input matrix if redundant nodes are present in
        the input matrix. Redundant referes to a node that has all the same
        data values as another node. The node which is lower in number is
        retained when a redundant node is found. Ex. If Node 34 has the same 
        data as Node 1. Node 34 will be removed from node volt matrix.
        ----- true_branches: The list of true branches is also an output, it 
        has redundant nodes replaced with lower node label number non-redundant 
        nodes. Ex. Imagine Node 34 has the same data as Node 1. Any branches in
        the true_branches matrix containing node 34, let's say a branch (7,34).
        The output true_branches will now list (7,1). If there's a branch from
        (1,34) it will have to be deleted since a branch (1,1) is a 
        self-referring branch. Self-referring branches are sometimes returned 
        by this function. Remove_redundant_branches.m removes self-referring 
        branches and should be called after this function. Therefore
        the true_branches variable should have the same size before and after
        this function call since no branches are removed. """

        number_observations = len(self.vmag_matrix[:,1])

        # We initialize a variable for the number of identical nodes found. 
        # This is useful for debugging purposes.
        number_identical_nodes = 0

        # We will cycle through the node_volt_matrix and see if we can find
        # redundant nodes. If so we will delete this node. Note the number of     
        # columns (i.e. nodes) is changing, this means we need to
        # evaluate this number every loop of the cycle.
        
        # We have to use a while loop since in Matlab for loops evaluate 
        # bounds at first run and we will change the size of our matrix that 
        # we iterate through, since we find identical pairs, we'll remove 
        # columns. 
        outer_loop_index = 0;

        matrix_of_identical_nodes = np.zeros(((self.num_buses -1),(2)))

        while (outer_loop_index < (len(self.vmag_matrix[1,:])-1)):
        # We want to evaluate whether one node is the same as another
        # column so we check starting with node i+1.
        # Re-intialize k.
            #print('outer loop index')
            #print(outer_loop_index)
            inner_loop_index = outer_loop_index + 1
            while (inner_loop_index <= (len(self.vmag_matrix[1,:]) -1)):
                #print('inner loop index')
                #print(inner_loop_index)
        
                # Check whether it is the same node.
                if (np.sum(self.vmag_matrix[:,outer_loop_index] == 
                           self.vmag_matrix[:,inner_loop_index]) == \
                            number_observations):
                    #print('here')
            
                    number_identical_nodes = number_identical_nodes + 1
                    matrix_of_identical_nodes[number_identical_nodes, :] = \
                                             outer_loop_index,inner_loop_index
                    # Now let's remove the identical nodes.
                    self.vmag_matrix = \
                        np.delete(self.vmag_matrix, inner_loop_index, 1)
            
            # Update the list of true branches.
            # Go through the rows of the true branch data set.
                    for g in range(0, self.num_buses -1):
                # Check if we find the second identical node-k in the row.
                        for h in range(0, 2):
                            if self.true_branches[g,h] == inner_loop_index:
                                self.true_branches[g,h] = outer_loop_index
                            elif self.true_branches[g,h] > inner_loop_index:
                    # Since I've just deleted a col, I also need to
                    # downshift all the values in the True Branch Data set 
                    # that were above this col. Ex. if i delete column 5 
                    # (node 5), i need to make nodes 6 appear as node 5 in 
                    # the true branch data.
                                self.true_branches[g,h] = \
                                                  self.true_branches[g,h] -1

        # If a column was not deleted continue iterating. We don't 
        # increment iterator if a col was deleted since now col k is a 
        # different col.
                else:
                    inner_loop_index = inner_loop_index +1
            outer_loop_index = outer_loop_index + 1
            
        # Update self.num_buses
        self.num_buses = len(self.vmag_matrix[1,:])
            
        
        
        
                             
                

        
        
        
        

            
            
            
        
    
