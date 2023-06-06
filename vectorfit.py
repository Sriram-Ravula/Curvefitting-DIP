import skrf as rf
import numpy as np
import time
import math
import os
import subprocess

"""
Demo for using vectorfit.py

The library will load a Touchstone S-Parameter file as ground truth,
and create a vector fit for it using some user-specified
freuquency points. The fitter automatically tunes the
vector fit model to use the best number of poles.

The basic use is

>>> fitter = vectorfit.VectorFitter(<touchstonefile>)
>>> vectorfit = fitter.vector_fit(<name>, <frequency samples>)
>>> fitted_network = vectorfit.fitted_network

Here fitted_network is a skrf.Network instance. This will
have the same frequencies as the source ground truth S-Parameter
network, but these have been computed using a vector fit model
with the user specified frequency samples.
"""

class FitError:
    """
    This class captures the error between a fitted network
    and the orignal network with the same frequency points.
    
    ground_truth: This is the true data, sampled from the
        full source (usually 1000 frequency points).
    fitted_network: This network results from invoking
        vector fit on the ground truth.
    
    There are many ways to do this, I'm just using
    the maximum error between the two networks over
    the frequency range.
    """
    def __init__(self, fitted_network, ground_truth):
        self.fitted_network = fitted_network
        self.ground_truth = ground_truth
        
        self.use_weighted_error = False
        self.error_weight_sigmoid_control = 10
        
        self._fill_errors()
    
    def _error_weight_function(self, error_value):
        """
        We can try weighting to deemphasize low magnitude errors.
        """
        if not self.use_weighted_error:
            return error_value
        return 1.0 / (1.0 + math.exp(-self.error_weight_sigmoid_control*(abs(error_value) - 0.5)))
    
    @staticmethod
    def _max_error_between_matrices(mat1, mat2, relative=False, weight_callback=None):
        if mat1.shape != mat2.shape:
            raise RuntimeError("Matrices are different shapes and cannot be compared.")

        max_diff = 0
        for val1, val2 in np.nditer([mat1, mat2]):
            complex_diff = val1-val2
            abs_diff = abs(complex_diff)
            abs_val1 = abs(val1)
            abs_val2 = abs(val2)
            if relative:
                abs_diff = abs_diff / max(abs_val1, abs_val2)
            if weight_callback is not None:
                abs_diff = weight_callback(abs_diff) * abs_diff
            if abs_diff > max_diff:
                max_diff = abs_diff
        return max_diff
    
    def _error_at_frequency_index(self, ff):
        fit_at_frequency = self.fitted_network.s[ff, :, :]
        truth_at_frequency = self.ground_truth.s[ff, :, :]
        err = FitError._max_error_between_matrices(
            fit_at_frequency, truth_at_frequency, weight_callback=self._error_weight_function)
        return err
        
    def _fill_errors(self):
        # Compute the difference at each frequency
        self.errors_by_frequency = [self._error_at_frequency_index(ff) for ff in range(len(self.fitted_network.f))]
    
    @property
    def max_error(self):
        return max(self.errors_by_frequency)
    
    @property
    def min_error(self):
        return min(self.errors_by_frequency)
    
    @property
    def ave_error(self):
        self.ave_error = sum(self.errors_by_frequency) / len(self.errors_by_frequency)
    
    @property
    def error(self):
        return self.max_error
            

class VectorFitResult:
    """
    This class just stores the vector fitted result. We really just need
    the fitted_network, but I didn't want to throw away the other data
    here (the error and vector fit model). I decided to return as a class
    instead of returning all the data in a tuple.
    
    fitted_network: skrf.Network instance with the S-Parameter network
        at the ground_truth (aka ground_truth, usually 1000) sample points, as prediced bytearray
        the vector fit model created from a subset sample.
    vector_fit_model: skrf.vectorFitting.VectorFitting instance with the
        full vector fit model. It's just kept for reference
    error_vs_sampled: FitError instance comparing fitted_network with the
        sampled network used as the source for the vector fit. This is
        knowable during the adaptive frequency sweep process, and is useful
        in guiding further sample points.
    error_vs_ground_truth: FitError instance comparing the fitted_network
        with the full ground_truth source. This not knowable in any functional
        algorithm, but gives the best measure of how the fit performs given
        perfect knowledge.
    """
    def __init__(self, name, fitted_network, vector_fit_model, error_vs_sampled, error_vs_ground_truth):
        # Vector Fit mode
        self.name = name
        self.fitted_network = fitted_network
        self.vector_fit_model = vector_fit_model
        self.error_vs_sampled = error_vs_sampled
        self.error_vs_ground_truth = error_vs_ground_truth
    
    @property
    def n_poles(self):
        return len(self.vector_fit_model.poles)
        

class VectorFitter:
    def __init__(self, touchstone_file, max_poles=50, converged_error_tol=2e-4,
                 number_of_failed_improvements_before_quit=2, verbose=True):
        self.touchstone_file = touchstone_file
        self.max_poles = max_poles
        self.converged_error_tol = converged_error_tol
        self.number_of_failed_improvements_before_quit = number_of_failed_improvements_before_quit
        self.verbose = verbose
        
        # Load the ground truth model. We'll use this
        # instead of simulating each requested frequency.
        try:
            self.ground_truth = rf.Network(touchstone_file)
        except:
            self.ground_truth = touchstone_file
    
    def _status_update(self, msg):
        if self.verbose:
            print(msg)
    
    @staticmethod
    def fill_fitted_network(frequencies, vector_fit):
        n_ports = vector_fit.network.s.shape[1]
        s = np.zeros((len(frequencies), n_ports, n_ports), dtype=complex)
        for p1 in range(n_ports):
            for p2 in range(n_ports):
                s[:,p1, p2] = vector_fit.get_model_response(p1, p2, frequencies)
        #A, B, C, D, E = vector_fit._get_ABCDE()
        #for ff, freq in enumerate(frequencies):
        #    s_at_frequency = vector_fit._get_s_from_ABCDE(freq, A, B, C, D, E)
        #    s[ff, :, :] = s_at_frequency
        return rf.Network(frequency=rf.Frequency.from_f(frequencies, unit="Hz"), s=s)
    
    def _sample_network(self, frequency_sample_points_in_hz):        
        frequency_instance = rf.Frequency.from_f(frequency_sample_points_in_hz, unit="Hz")
        new_network = self.ground_truth.interpolate(frequency_instance)
        return new_network
    
    def _is_better(self, next_fit, previous_best):
        """
        We could implement some complex heuristics here to captures
        when the additional poles aren't adding anything significant
        to the fit. I'm doing something simple here.
        """
        if previous_best is None:
            return True
        diff = previous_best.error_vs_sampled.error - next_fit.error_vs_sampled.error
        return diff > self.converged_error_tol
    
    @staticmethod
    def _maximum_poles_mathematical_limit(frequency_sample_points_in_hz):

        # The maximum number of poles is limited by the number
        # of frequency points. I'm sure it can be found analytically,
        # but I found this analytically.
        num_frequency_points = len(frequency_sample_points_in_hz)

        # The behavior is a little weird at low counts, so I'm just overriding
        low_count_values = {0: 0, 2: 0, 3: 1, 4: 0}
        if num_frequency_points in low_count_values.keys():
            return low_count_values[num_frequency_points]

        # Experimentally, I've found the maximum to be roughly half the number
        # of frequency points minus two, with slightly different limits for
        # even and odd counts.
        if num_frequency_points % 2 == 0:
            return num_frequency_points // 2 - 2
        else:
            return (num_frequency_points+1) // 2 - 2
    
    def vector_fit(self, name, frequency_sample_points_in_hz):
        start_time = time.time()
        
        # Sample the orignal at the specified frequency points
        sampled_network = self._sample_network(frequency_sample_points_in_hz)
        
        # Vector fit will only work up to n_poles ~= half the number
        # of frequency points. We also impose a maximum number of poles
        # becuase we have to stop somewhere.
        number_of_poles_mathematical_limit = VectorFitter._maximum_poles_mathematical_limit(frequency_sample_points_in_hz)
        maximum_poles = min(self.max_poles, number_of_poles_mathematical_limit)
        
        # Find the best fit given the available models
        failed_improvements = 0
        n_evals = 0
        best_fit = None
        for n_poles in range(1, maximum_poles):
            n_evals += 1
            self._status_update(f"Performing fit with {n_poles} poles.")
            
            # Perform the vector fit
            vf = rf.VectorFitting(sampled_network)
            vf.vector_fit(n_poles_cmplx=n_poles)
            self._status_update(f"  Fitting took {vf.wall_clock_time} seconds.")
            
            # Get the fitted network at the sampled points
            self._status_update(f"  Filling sampled network.")
            sampled_network_fit = VectorFitter.fill_fitted_network(frequency_sample_points_in_hz, vf)
            
            # Get the fitted network at the full sweep points
            self._status_update(f"  Filling full sweep network.")
            full_sweep_network_fit = VectorFitter.fill_fitted_network(self.ground_truth.f, vf)
            
            # Compute the errors
            self._status_update(f"  Computing error.")
            error_vs_sampled = FitError(sampled_network_fit, sampled_network)
            error_vs_ground_truth = FitError(full_sweep_network_fit, self.ground_truth)
            
            
            fit = VectorFitResult(name, full_sweep_network_fit, vf, error_vs_sampled, error_vs_ground_truth)
           
            if self._is_better(fit, best_fit):
                failed_improvements = 0
                if best_fit is not None:
                    self._status_update(f"  Fit with {fit.n_poles} poles better than {best_fit.n_poles} " +
                                        f"(err {fit.error_vs_sampled.error} vs {best_fit.error_vs_sampled.error})")
                best_fit = fit
            else:
                self._status_update(f"  Fit with {fit.n_poles} was not subtantially better than {best_fit.n_poles} " +
                                    f"(err {fit.error_vs_sampled.error} vs {best_fit.error_vs_sampled.error})")
                failed_improvements += 1
            
            # If we're failing to improve the fit, we can quit instead
            # of wasting time on higher order fits.
            if failed_improvements > self.number_of_failed_improvements_before_quit:
                break
        
        end_time = time.time()
        self._status_update(f"Fit completed in {end_time - start_time}s, using {n_evals}, resulting" +
                            f" in a fit with {best_fit.n_poles} and error={best_fit.error_vs_sampled.error} vs samples" +
                            f" and error={best_fit.error_vs_ground_truth.error} vs ground truth.")
        return best_fit

def _delete_files(files):
    for file in files:
        try:
            os.remove(file)
        except OSError:
            pass


class StaticVectorFit:

    @staticmethod
    def sample_network(source_file, frequency_sample_points_in_hz):
        full_network = rf.Network(source_file)
        frequency_instance = rf.Frequency.from_f(frequency_sample_points_in_hz, unit="Hz")
        new_network = full_network.interpolate(frequency_instance)
        return new_network
    
    @staticmethod
    def save_sampled_network(source_file, save_file, frequency_sample_points_in_hz):
        sampled = SiemensVectorFit.sample_network(source_file, frequency_sample_points_in_hz)
        _, src_ext = os.path.splitext(source_file)
        base, _ = os.path.splitext(save_file)
        save_file_corrected = base + src_ext
        _delete_files([save_file_corrected])
        sampled.write_touchstone(save_file_corrected)
        return save_file_corrected

class ScikitVectorFit(StaticVectorFit):
    
    @staticmethod
    def vector_fit(observations_s_parameters_file, desired_frequency_points):
        # Create a vector fit model using the saved data
        fitter = VectorFitter(observations_s_parameters_file)
        vf = fitter.vector_fit("SciKit Fit", fitter.ground_truth.f)
        
        # Interpret the vf model at the requested points
        fitted_network = VectorFitter.fill_fitted_network(desired_frequency_points, vf.vector_fit_model)
        
        # Return the fitted network
        return fitted_network
    
    @staticmethod
    def vector_fit_samples(ground_truth_s_parameters_file, sample_frequency_points, desired_frequency_points,
                           observations_file="observations"):
        obs = StaticVectorFit.save_sampled_network(ground_truth_s_parameters_file, observations_file, sample_frequency_points)
        return ScikitVectorFit.vector_fit(obs, desired_frequency_points)


class SiemensVectorFit(StaticVectorFit):
        
    @staticmethod
    def _write_solver_options(desired_frequency_points, filename):
        with open(filename, 'w') as fout:
            fout.write('<SolverOptions>\n')
            fout.write('  <Header Version="6.13.0" />\n')
            fout.write('  <Frequency  Type="SCATTERED" >\n')
            for frequency in desired_frequency_points:
                fout.write(f'    <Point Value="{frequency}"/>\n')
            fout.write('    <Options FastSweep="Yes"  />\n')
            fout.write('    <Sliders AFSConvergenceSlider="0" AFSMagnifySlider="0" AFSIntervalErrorSlider="0" />\n')
            fout.write('    <Advanced MinSplineFreqPointCnt="8" Order="4" Threshold="0.001" MaxAdaptiveSweepPoints="500" />\n')
            fout.write('  </Frequency>\n')
            fout.write('</SolverOptions>\n')
    
    @staticmethod
    def vector_fit(observations_s_parameters_file, desired_frequency_points, path_to_hlasconsole, optionsfile="SolverOptions.opt"):
        
        base, ext = os.path.splitext(observations_s_parameters_file)
        
        # Delete the old files
        _delete_files([f"Fitter_Fit_{base}" + ext,
                       f"Fitter_HLAS_{base}" + ext,
                       f"Fitter_HLAS2_{base}" + ext,
                       f"Fitter_Spline_{base}" + ext,
                       optionsfile])
                       
        # Create a solver options file using these frequency points
        SiemensVectorFit._write_solver_options(desired_frequency_points, optionsfile)
        
        # Run the Siemens/HLAS fitter
        executable = os.path.join(path_to_hlasconsole, "hlasConsole")
        subprocess.run([executable, "-tool", "test", "testvf", observations_s_parameters_file, optionsfile])
        
        # Return the fitted network
        return rf.Network(f"Fitter_HLAS_{base}" + ext)
    
    @staticmethod
    def vector_fit_samples(ground_truth_s_parameters_file, sample_frequency_points,
                           desired_frequency_points, path_to_hlasconsole,
                           observations_file="observations", optionsfile="SolverOptions.opt"):
        obs = StaticVectorFit.save_sampled_network(ground_truth_s_parameters_file, observations_file, sample_frequency_points)
        return SiemensVectorFit.vector_fit(obs, desired_frequency_points, path_to_hlasconsole, optionsfile)

