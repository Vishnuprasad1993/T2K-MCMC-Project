
'''
Simple example set of functions to show how to read a MaCh3 output file
'''

# Lets us use ROOT files
import uproot
# This is me being pedantic, good python should have annotations/type hints to let the user know if they're
# doing the right thing
from typing import Any
# Python's standard plotting library
from matplotlib import pyplot as plt

#lets us save to pdf
from matplotlib.backends.backend_pdf import PdfPages
# The array format we're using
import pandas as pd
import numpy as np
from scipy.stats import norm
import math

# Autocorrelation plotter
#import statsmodels.api as sm

'''
Small toolkit for opening and plotting ROOT files
 -> open_root_file : Opens a ROOT file and outputs the TTree (table of variables)
 -> select_ttree_variables : Processes TTree by removing any unwanted variables and converts it into a pandas dataframe
 -> plot_parameter_traces : Plots traces for all the parameters you've cut out
'''


def open_root_file(file_name: str, ttree_name: str="osc_posteriors")->Any: 
    '''
    Function to open ROOT files

    inputs : 
        -> File name : name of file (remember to include the full file path!)

        optional:

        -> TTree Name : name the MCMC is stored in (defaults to osc_posteriors)

    outputs : 
        -> ROOT TTree object containing posteriors
    '''

    # Open our ROOT file
    posterior_tree =  uproot.open(f"{file_name}:{ttree_name}") # opens file

    # Check the file we're trying to use actually exists!
    if not posterior_tree:
        raise IOError(f"Error couldn't open {file_name}:{ttree_name}")

    # returns the file object [this is kind of bad practice but oh well]
    return posterior_tree

def select_ttree_variables(posterior_ttree: Any, interesting_variables: list[str]=["theta23", "theta13", "theta12", "dm23", "dm12", "dcp"])->pd.DataFrame:
    '''
    Selects branches you want to use in the TTree from a list of names

    inputs : 
        -> posterior_ttree : A ROOT TTree (output from open_root_file)

        optional
            -> interesting_variables : List of variables we care about (defaults to oscillation parameters)
            -> table_format : What do we want our final table to be (defaults to pandas)

    outputs
        -> Processed TTree as nice table
    '''
    # Again let's program defensively and break if we're given nonsense!
    if not posterior_ttree:
        raise IOError("Error passed non-existent ttree object")
    
    # Selecting the variables we want is super simple!
    posterior_table = posterior_ttree.arrays(interesting_variables, library="pd")

    posterior_ttree.close() # Let's close our file since we've got everything we need from it!

    # Returns our TTree as a nice iterable table
    return posterior_table

def plot_parameter_traces(input_dataframe: pd.DataFrame, output_file_name: str="parameter_traces")->Any:
    '''
    Plots parameter traces for ALL parameters in an pandas dataframe

    inputs:
        input_array : output of select_ttree_variables

        optional
        -> output_file_name : name of pdf file to output to

    returns:
        None
    '''
    
    
    # selecting the first row from the dataframe, converting to a list, then return
    #return input_dataframe.iloc[0].tolist()
    return np.array(input_dataframe.iloc[0])
    '''
    # make sure we're saving to a pdf
    if(output_file_name[-4:]!=".pdf"):
        output_file_name+=".pdf"

    # open our output file [multi-page PDF]
    with PdfPages(output_file_name) as pdf:
        
        # Loop over columns in our data frame
        for col in input_dataframe.columns:
            # Makes a figure to draw plots to
            fig = plt.figure()
            # Setup up the plot title + axis labels
            plt.title(f"{col} trace")
            plt.ylabel(f"{col}")
            plt.xlabel("Step")
            # Make the plot
            plt.grid()
            plt.plot(input_dataframe[col], linewidth=0.05, color='black')
            # Save to the pdf
            pdf.savefig(fig)
            # Clear the plotting canvas
            plt.close()
    ''' 

            
# identifying regeneration times
def Regeneration_time (input_chain, start, step_size_arr)->Any:
    #separator = '\n----------------------------------------'
    #print(f"Starting point (Initial state) : {start}", separator) #Printing the starting point of the chain
    #print(f"Regeneration width using = {step_size_arr}")
    #total_rows = len(input_chain)
    #print(input_chain['theta23'])
    #number_of_regeneration = 0
    '''
    input_chain = input_chain.drop(input_chain.index[0])
    input_chain = input_chain - start.squeeze()
    condition1 = abs(input_chain['theta23']) <= step_size_arr[0]
    condition2 = abs(input_chain['theta13']) <= step_size_arr[1]
    condition3 = abs(input_chain['theta12']) <= step_size_arr[2] 
    condition4 = abs(input_chain['dm23']) <= step_size_arr[3]
    condition5 = abs(input_chain['dm12']) <= step_size_arr[4]
    condition6 = abs(input_chain['dcp']) <= step_size_arr[5]
    #print(condition1)
    
    mask = condition1 & condition2 & condition3 & condition4 & condition5 & condition6
    #print(mask)
    #print(input_chain)
    number_of_regeneration_states = mask.sum()
    mask_dict = mask.to_dict()
    true_values_dict = {key: value for key, value in mask_dict.items() if value}
    regeneration_states = list(true_values_dict.keys())
    
   
    #increased step_size between regeneration states
    new_regeneration_states = [regeneration_states[i] for i, _ in enumerate(regeneration_states) if i % 10 == 0]
    #new_regeneration_states = [new_regeneration_states[i] for i, _ in enumerate(new_regeneration_states) 
                               #if new_regeneration_states[i]-new_regeneration_states[i-1] > 10]
    #print(true_values_dict)
    print("regeneration_state", new_regeneration_states)
    steps_to_nearest_regeneration_state = [new_regeneration_states[i] - new_regeneration_states[i-1] for i in range(1, len(new_regeneration_states))] 
                                           #if new_regeneration_states[i] - new_regeneration_states[i-1] > 20000] 
                                          
    print("steps: ", steps_to_nearest_regeneration_state)
    return number_of_regeneration_states, steps_to_nearest_regeneration_state
    '''
    r = math.sqrt((step_size_arr[0])**2 + (step_size_arr[1])**2 + (step_size_arr[2])**2 + (step_size_arr[3])**2 + (step_size_arr[4])**2
                   + (step_size_arr[5])**2)
    print("r is", r)
     #extract values in every row of the dataframe and make a list
    values_list = [input_chain.iloc[i].values for i in range(len(input_chain))]
    print(values_list)
    result = []
    # Iterate over each array in values_list starting from the second array
    for array in values_list[1:]:
        list1 = []
        # Compare corresponding elements of the current array with the initial state
        for val_init, val_curr in zip(start, array):
             #finding the Euclidean distance matrix
             list1 += [(val_curr - val_init)**2]
             
        #print(list1)
        result1 = math.sqrt(sum(list1))
        #print(result1)
        result += [result1]
    print("result is", result) # result is a list of Euclidean distances between initial state and all other states
    number_of_regeneration_states = 0
    
    regeneration_states = []
    for x, value in enumerate(result):
        if abs(value) <= r:
            #print("The regeneration state is reached")
            number_of_regeneration_states += 1
            regeneration_states += [x + 1]
    #print(number_of_regeneration_states)
    #print(regeneration_states)
    #increased step_size between regeneration states
    new_regeneration_states = [regeneration_states[i] for i, _ in enumerate(regeneration_states) if i % 10 == 0]
    #print("regeneration_state", new_regeneration_states)
    steps_to_nearest_regeneration_state = [new_regeneration_states[i] - new_regeneration_states[i-1] for i in range(1, len(new_regeneration_states))]
    #print("steps: ", steps_to_nearest_regeneration_state)
    return number_of_regeneration_states, steps_to_nearest_regeneration_state
    
def vary_regeneration_width(input_chain, start, step_size_arr):
    
    step_size_arr = np.array(step_size_arr)
    
    scale_arr = []
    steps_to_regeneration = []
    number_of_regeneration = []
    #scale_arr = np.empty((1,0), dtype=int)
    for scale in range(1, 30):
        scale_arr = scale_arr + [scale]
        number_of_regeneration_states, steps_to_nearest_regeneration_state = Regeneration_time(input_chain, start, scale*step_size_arr)       
        steps_to_regeneration = steps_to_regeneration + [steps_to_nearest_regeneration_state]
        number_of_regeneration = number_of_regeneration + [number_of_regeneration_states]
       
    print("Steps to regeneration: ", steps_to_regeneration)
    print("number of regeneration: ", number_of_regeneration)
        #print("Scale_arr:", scale_arr)
    
    plot_number_of_regeneration_states_vs_width(number_of_regeneration, scale_arr)
    plot_frequency_vs_regeneration_states (steps_to_regeneration, scale_arr)

def plot_frequency_vs_regeneration_states(steps_to_regeneration, scale_arr):
    with PdfPages('Regeneration_state_frequency_plots.pdf') as pdf:
         for label, values in zip(scale_arr, steps_to_regeneration):
             #considering only a small portion of data
             values = [i for i in values if 11 <= i <= 1000]
             plt.figure()
                # CHANGE THE 100 THIS IS A BAD IDEA!
             plt.hist(values, bins=20, color='b')
        
             plt.title(f'Frequency Plot - Scale {label}')
             plt.xlabel('Steps to nearest regeneration state')
             plt.ylabel('Frequency')
             pdf.savefig()

             plt.close()
    
def plot_number_of_regeneration_states_vs_width(number_of_regeneration, scale_arr):
    plt.rcParams.update({'figure.figsize':(10,8),'figure.dpi':100})
    plt.plot(scale_arr, number_of_regeneration)
    plt.title('Number of regeneration states vs Width Graph')
    plt.xlabel('Scale')
    plt.ylabel('Number of regeneration states')
    plt.savefig('number of regeneration states vs width graph.pdf', format = 'pdf')
    # plt.show()
  

'''
def plot_parameter_autocorrelations(input_dataframe: pd.DataFrame, output_file_name: str="parameter_autocorrelations", total_lags=10000)->None:
    
    Plots parameter traces for ALL parameters in an pandas dataframe

    inputs:
        input_array : output of select_ttree_variables

        optional
        -> output_file_name : name of pdf file to output to
        -> total_lags : How much lag do you want?

    returns:
        None
    

    make sure we're saving to a pdf
    if(output_file_name[-4:]!=".pdf"):
        output_file_name+=".pdf"

    with PdfPages(output_file_name) as pdf:
        # loop over columns in DF
        for col in input_dataframe.columns:
            # Get the ACF
            acf = sm.tsa.acf(input_dataframe[col], nlags=total_lags)
            # Makes a figure to draw plots to
            fig = plt.figure()
            plt.plot(acf)
            # Setup up the plot title + axis labels
            plt.title(f"{col} ACF")
            plt.ylabel("Autocorrelation")
            plt.xlabel("Lag")
            # Make the plot
            plt.grid()
            # Save to the pdf
            pdf.savefig(fig)
            # Clear the plotting canvas
            plt.close()

'''

if __name__=="__main__":
    # Running in the __main__ scope means this bit only gets run if we run chain_reader.py

    # Open our root file
    POSTERIOR = open_root_file("../inputs/markov_chains/oscpar_only_datafit.root")

    #Let's just have our parameter names spelt out explicitly 
    OSCILLATION_PARAMETERS = ["theta23", "theta13", "theta12", "dm23", "dm12", "dcp"]
    
    # pandas dataframe containing our chain output
    POSTERIOR_TABLE = select_ttree_variables(POSTERIOR, OSCILLATION_PARAMETERS)
    #print(POSTERIOR_TABLE)
    
    
    '''
    Step sizes used for each parameter
    theta23: 0.045×0.021   = 9.45 x 10^-4
    theta13: 0.225×7×10^-4 = 1.575 x 10^-4
    theta12: 0.045×0.013   = 5.85 x 10^-4
    dm23: 0.045×3.4×10^-5  = 0.153 x 10^-5
    dm12: 0.045×1.8×10^-6  = 0.081 x 10^-6
    dcp: 0.045×6.28        = 0.2826
    '''
    step_size_arr = [9.45e-4, 1.575e-4, 5.85e-4, 0.153e-5, 0.081e-6, 0.2826] 
    
    # makes some nice plots!
    # start , (Starting point of chain) a list of elements containing in the first row of POSTERIOR_TABLE
    start = plot_parameter_traces(POSTERIOR_TABLE)
    
    vary_regeneration_width(POSTERIOR_TABLE, start, step_size_arr)
    
    
   