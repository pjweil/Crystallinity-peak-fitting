import numpy as np
import os
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.integrate import quad


mca_number = '1' # Insert number of polymer number here, for example, '1'

# Define functions to make gaussian peaks
def gaussian(x, height, center, width):
    return height * np.exp(-(x - center)**2 / (2 * width**2))

def multi_gaussian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        y += gaussian(x, params[i], params[i+1], params[i+2])
    return y

def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


# Navigate to the directory where the .dat files are located
os.chdir('/Users/phoebeweil/Desktop/Thesis/waxs/WAXSdata_rename')

# Read and average the .dat files
files_to_average = [f'MCA{mca_number}_1.dat', f'MCA{mca_number}_2.dat', f'MCA{mca_number}_3.dat'] # Replace these with file names as needed
data_arrays = []

for file_name in files_to_average:
    data = np.loadtxt(file_name, usecols=(1,), unpack=True)
    data_arrays.append(data)

average_data = np.mean(data_arrays, axis=0)
x_data = np.loadtxt(f'MCA{mca_number}_1.dat', usecols=(0,), unpack=True)

# Subtract the blank
blank_data = np.loadtxt('blank_1.dat', usecols=(1,), unpack=True)
subtracted_data = average_data - blank_data

# Perform a baseline correction
# Subtract a straight line along the bottom of the peaks, between x=0.15 and x=1.5
# Find indices within the specified x ranges
indices_015_to_06 = (x_data >= 0.15) & (x_data <= 0.6)
indices_1_to_15 = (x_data >= 1) & (x_data <= 1.5)

# Find minimum y values within these x ranges
y_min_015_to_06 = np.min(subtracted_data[indices_015_to_06])
y_min_1_to_15 = np.min(subtracted_data[indices_1_to_15])

# Find the x values corresponding to these minimum y values
x_min_015_to_06 = x_data[indices_015_to_06][np.argmin(subtracted_data[indices_015_to_06])]
x_min_1_to_15 = x_data[indices_1_to_15][np.argmin(subtracted_data[indices_1_to_15])]

# Calculate the slope (m) and intercept (b) of the line connecting these minimum points
m = (y_min_1_to_15 - y_min_015_to_06) / (x_min_1_to_15 - x_min_015_to_06)
b = y_min_015_to_06 - m * x_min_015_to_06

# Subtract the linear background from the average data
linear_background = m * x_data + b
subtracted_data = subtracted_data - linear_background

# Update the filtered data for peak fitting
indices_of_interest = (x_data >= 0.15) & (x_data <= 1.5)
x_data_filtered = x_data[indices_of_interest]
subtracted_data_filtered = subtracted_data[indices_of_interest]


# Identified peak values, initial guesses depend on the diol used
# Uncomment lines based on the polymer being studied

# Good initial guesses for ethylene glycol polymers
# peak_centers = [0.4317, 0.7457, 0.7966, 0.9005, 1.0475, 1.2146, 0.9099]
# peak_widths = [0.02, 0.0452, 0.01, 0.0625, 0.00512, 0.02, 0.5] #initial guesses for peak widths
# max_intensity = [0.0235, 0.1302, 0.02, 0.0463, 0.0535, 0.0380, 0.17] #initial guesses for max intensities

# # Good initial guesses for 1,3-propanediol polymers
# peak_centers = [0.457, 0.7433, 0.8431, 1, 1.04, 1.12429, 1.283, 0.954]
# peak_widths = [0.04, 0.0009, 0.000446, 0.00512, 0.00512,0.0512, 0.00272, 1] #initial guesses for peak widths
# max_intensity = [0.011, 0.1082, 0.009, 0.0439, 0.0439, 0.0439, 0.0668, 0.0476]

# Good initial guesses for 1,4-butanediol polymers
# peak_centers = [0.47, 0.7976, 0.8431, 0.9783, 0.9497, 1.1114, 1.05]
# peak_widths = [0.01, 0.0009, 0.0005, 0.0512, 0.00272, 0.0418, 0.01] #initial guesses for peak widths
# max_intensity = [0.04, 0.0341, 0.0459, 0.1238, 0.0668, 0.1444, 0.0410] #initial guesses for max intensities

# Good initial guesses for 2,2-dimethyl-1,3-propanediol polymers
# peak_centers = [0.3943, 0.4279, 0.72, 0.78, 0.94, 0.98, 0.9]
# peak_widths = [0.002, 0.002, 0.04, 0.08, 0.005, 0.005, 0.5] #initial guesses for peak widths
# max_intensity = [0.0375, 0.0571, 0.0914, 0.1004, 0.01, 0.038, 0.05]

# # Good initial guesses for 1,4-cyclohexane dimethanol polymers
peak_centers = [0.43, 0.7457, 0.7966, 0.9005, 0.9783, 0.8776, 1.4]
peak_widths = [0.06, 0.0452, 0.05, 0.0625, 0.0512, 1, 0.02] #initial guesses for peak widths
max_intensity = [0.0237, 0.1302, 0.1393, 0.0463, 0.1238, 0.1941, 0.0166] #initial guesses for max intensities


# Initialize
initial_guesses = []
lower_bounds = []
upper_bounds = []

for i, center in enumerate(peak_centers):
    initial_guess = [max_intensity[i], center, peak_widths[i]]
    initial_guesses.extend(initial_guess)
    lower_bounds.extend([0, center - 0.05, 0])  # Ensure positive height, allow some deviation in center, positive width
    upper_bounds.extend([np.inf, center + 0.05, np.inf])  # No upper limit on height, slight deviation in center, no upper limit on width


# Iterative peak fitting
rmse = np.inf
threshold=0.007
iteration_limit = 500
current_iteration = 0

while rmse > threshold and current_iteration < iteration_limit:
    params, cov = curve_fit(multi_gaussian, x_data_filtered, subtracted_data_filtered, p0=initial_guesses, bounds=(lower_bounds, upper_bounds))
    fitted_curve = multi_gaussian(x_data_filtered, *params)
    rmse = calculate_rmse(subtracted_data_filtered, fitted_curve)
    
    print(f"Iteration {current_iteration}: RMSE = {rmse}")
    
    if rmse > threshold:
        for i in range(2, len(initial_guesses), 3):
            initial_guesses[i] *= np.random.uniform(0.95, 1.05)
    
    current_iteration += 1

if current_iteration == iteration_limit:
    print("Warning: Reached iteration limit without converging to desired RMSE.")


# Plot the averaged and subtracted data
plt.figure(figsize=(10, 6))
plt.plot(x_data_filtered, subtracted_data_filtered, 'k-', label='Raw Data')

# Generate and plot each Gaussian peak using the fitted parameters
for i in range(0, len(params), 3):
    peak_x = np.linspace(x_data_filtered.min(), x_data_filtered.max(), 973)
    peak_y = gaussian(peak_x, params[i], params[i+1], params[i+2])
    plt.plot(peak_x, peak_y, '--', label=f'Gaussian {i//3+1}')

# Prepare the legend
mca_legend = {
    '1': "1,4 cyclohexane dimethanol and succinic acid",
    '2': "1,4 cyclohexane dimethanol and glutaric acid",
    '3': "1,4 cyclohexane dimethanol and adipic acid",
    '4': "1,4 cyclohexane dimethanol and pimelic acid",
    '5': "1,4 cyclohexane dimethanol and suberic acid",
    '6': "1,4 cyclohexane dimethanol and azeliac acid",
    '7': "1,4 cyclohexane dimethanol and sebacic acid",
    '8': "1,4 cyclohexane dimethanol and isophthalic acid",
    '9': "1,4 cyclohexane dimethanol and terephthalic acid",
    '10': "1,4 cyclohexane dimethanol and diglycolic acid",
    '11': "1,4 cyclohexane dimethanol and 1,4-cyclohexane dicarboxylic acid",
    '12': "Ethylene glycol and succinic acid",
    '13': "Ethylene glycol and glutaric acid",
    '14': "Ethylene glycol and adipic acid",
    '15': "Ethylene glycol and pimelic acid",
    '16': "Ethylene glycol and suberic acid",
    '17': "Ethylene glycol and azeliac acid",
    '18': "Ethylene glycol and sebacic acid",
    '19': "Ethylene glycol and isophthalic acid",
    '20': "Ethylene glycol and terephthalic acid",
    '21': "Ethylene glycol and diglycolic acid",
    '22': "Ethylene glycol and 1,4-cyclohexane dicarboxylic acid",
    '23': "2,2 dimethyl 1,3 propanediol and succinic acid",
    '24': "2,2 dimethyl 1,3 propanediol and glutaric acid",
    '25': "2,2 dimethyl 1,3 propanediol and adipic acid",
    '26': "2,2 dimethyl 1,3 propanediol and pimelic acid",
    '27': "2,2 dimethyl 1,3 propanediol and suberic acid",
    '28': "2,2 dimethyl 1,3 propanediol and azeliac acid",
    '29': "2,2 dimethyl 1,3 propanediol and sebacic acid",
    '30': "2,2 dimethyl 1,3 propanediol and isophthalic acid",
    '31': "2,2 dimethyl 1,3 propanediol and terephthalic acid",
    '32': "2,2 dimethyl 1,3 propanediol and diglycolic acid",
    '33': "2,2 dimethyl 1,3 propanediol and 1,4-cyclohexane dicarboxylic acid",
    '34': "1,4 butanediol and succinic acid",
    '35': "1,4 butanediol and glutaric acid",
    '36': "1,4 butanediol and adipic acid",
    '37': "1,4 butanediol and pimelic acid",
    '38': "1,4 butanediol and suberic acid",
    '39': "1,4 butanediol and azeliac acid",
    '40': "1,4 butanediol and sebacic acid",
    '41': "1,4 butanediol and isophthalic acid",
    '42': "1,4 butanediol and terephthalic acid",
    '43': "1,4 butanediol and diglycolic acid",
    '44': "1,4 butanediol and 1,4-cyclohexane dicarboxylic acid",
    '45': "1,3 propanediol and succinic acid",
    '46': "1,3 propanediol and glutaric acid",
    '47': "1,3 propanediol and adipic acid",
    '48': "1,3 propanediol and pimelic acid",
    '49': "1,3 propanediol and suberic acid",
    '50': "1,3 propanediol and azeliac acid",
    '51': "1,3 propanediol and sebacic acid",
    '52': "1,3 propanediol and isophthalic acid",
    '53': "1,3 propanediol and terephthalic acid",
    '54': "1,3 propanediol and diglycolic acid",
    '55': "1,3 propanediol and 1,4-cyclohexane dicarboxylic acid"
}

diol_diacid_combo = mca_legend[mca_number]

# Generate and plot the sum of all Gaussians
fitted_curve = multi_gaussian(x_data_filtered, *params)
plt.plot(x_data_filtered, fitted_curve, 'r-', linewidth=2, label='Sum of Gaussians')
plt.xlabel('q (inverse Angstroms)')
plt.ylabel('Intensity')
plt.title(f'WAXS Peak fitting: {diol_diacid_combo}')
plt.legend()
plt.show()


# Now, we get percent crystallinity from these peaks

# Function to calculate the full width half max (FWHM) of a Gaussian peak
def calculate_fwhm(width):
    return 2 * np.sqrt(2 * np.log(2)) * width

# Function to integrate a Gaussian peak
def integrate_gaussian(params):
    height, center, width = params
    integral, _ = quad(gaussian, -np.inf, np.inf, args=(height, center, width))
    return integral

# Calculate FWHM for each peak and their integrals
fwhms = []
integrals = []
for i in range(0, len(params), 3):
    fwhm = calculate_fwhm(params[i+2])
    fwhms.append(fwhm)
    integral = integrate_gaussian(params[i:i+3])
    integrals.append(integral)

# Find the largest FWHM, this peak is the "amorphous halo"
max_fwhm = max(fwhms)

# Select indices of all peaks with FWHM less than the maximum FWHM, these are the crystalline peaks
indices_of_peaks_less_than_max_fwhm = [i for i, fwhm in enumerate(fwhms) if fwhm < max_fwhm]

# Sum the areas of these peaks
sum_of_areas = sum(integrals[i] for i in indices_of_peaks_less_than_max_fwhm)

# Calculate the total area under the subtracted data curve
total_area, _ = quad(np.interp, x_data_filtered.min(), x_data_filtered.max(), args=(x_data_filtered, subtracted_data_filtered))

# Calculate the ratio
ratio = sum_of_areas / total_area

# print("FWHM of each peak:", fwhms)
# print(f"Indices of peaks with FWHM less than the maximum FWHM: {indices_of_peaks_less_than_max_fwhm}")
# print("Sum of areas of peaks with FWHM less than the maximum FWHM:", sum_of_areas)
# print("Total area under the MCA1 averaged and subtracted graph:", total_area)
print("Ratio:", ratio)


