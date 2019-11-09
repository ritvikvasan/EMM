import numpy as np
import matplotlib.pyplot as plt
import imageio
import skimage.filters
import scipy.io as sio
import similaritymeasures

class Symmetricize:
    def read_mat_file(path):
        """Extracts fit and outerfit data from .mat files

        Parameters
        ----------
        path : str
            The relative path to the .mat file.
        Returns
        -------
        dict
            A dictionary with the keys "x_fit", "y_fit", and "x_outerfit", "y_outerfit".
        """
        mat_contents = sio.loadmat(path)
        return {'x_fit': mat_contents['fit'][:,0], 'y_fit': mat_contents['fit'][:, 1],
                'x_outerfit': mat_contents['outerfit'][:, 0], 'y_outerfit': mat_contents['outerfit'][:, 1]}

    def make_symmetric(x_fit_data, y_fit_data):
        """Transforms non-symmetrical curvess into symmetrical ones

        Parameters
        ----------
        x_fit_data : numpy.ndarray
            An array containing all the x coordinates for fit data
        y_fit_data : numpy.ndarray
            An array containing all the y coordinates for fit data
        Returns
        -------
        numpy.ndarray, numpy.ndarray
            Arrays containing the symmetricized x and y coordinates
        """
        new_x_fit_data = np.absolute(x_fit_data) #that way we can calculate average x coord

        x_fit_data_transformed = np.zeros(len(new_x_fit_data)) #initialize resulting arrays
        y_fit_data_transformed = np.zeros(len(y_fit_data))

        x_start = np.argmin(np.absolute(0 - new_x_fit_data)) #what index has the value closest to 0 (so we know if the shape is skewed left/right)
        dist_from_mid = abs(len(new_x_fit_data) // 2 - x_start) #how far x_start is away from the actual middle of the array (used to shift resulting graph)

        #since x_start will rarely be exactly at middle of array, below expression ensures against arrayindexoutofbounds.
        for i in range(x_start if x_start <= (len(x_fit_data_transformed) // 2) else len(x_fit_data_transformed) - x_start):
            #since we want resulting figure to be perfectly symmetrical, we start from the middle of the new array and every value to left and right are the same
            x_fit_data_transformed[len(x_fit_data_transformed) // 2 - i] = - (new_x_fit_data[x_start - i] + new_x_fit_data[x_start + i]) / 2
            x_fit_data_transformed[len(x_fit_data_transformed) // 2 + i] = (new_x_fit_data[x_start - i] + new_x_fit_data[x_start + i]) / 2

        for i in range(len(y_fit_data_transformed) // 2 + 1):
            y_fit_data_transformed[i] = (y_fit_data[i] + y_fit_data[-1 - i]) / 2
            y_fit_data_transformed[-1 - i] = (y_fit_data[i] + y_fit_data[-1 - i]) / 2

        #after the above loops, there are some values we didn't take into account in the calculations, so we do them here
        leftover_data = (new_x_fit_data[0:dist_from_mid + 1] + new_x_fit_data[len(new_x_fit_data) - dist_from_mid - 1:]) / 2

        x_fit_data_transformed[0:dist_from_mid + 1] = - np.flip(leftover_data) #the first few data
        x_fit_data_transformed[len(new_x_fit_data) - dist_from_mid - 1:] = leftover_data #the last few datapoints
        #NOTE: the resulting array is not PERFECTLY symmetrical; sometimes it can be off by one x value
        return x_fit_data_transformed, y_fit_data_transformed

    def rotate_figure(x, y):
        """ Performs PCA to determine major/minor axes and rotates figure upright according to major axis

        Parameters
        ----------
        x : numpy.ndarray
            An array containing x coordinates of fit data
        y : numpy.ndarray
            An array containing y coordinates of fit data
        Returns
        -------
        numpy.ndarray, numpy.ndarray
            Arrays containing rotated x and y coordinates
        """
        x -= np.mean(x)
        y -= np.mean(y)
        coords = np.vstack([x, y])

        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)

        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
        x_v2, y_v2 = evecs[:, sort_indices[1]]

        scale = 20
        plt.plot([x_v1*-scale*2, x_v1*scale*2], [y_v1*-scale*2, y_v1*scale*2], color='red') #plots the major axis
        plt.plot([x_v2*-scale, x_v2*scale], [y_v2*-scale, y_v2*scale], color='blue') #plots the minor axis
        plt.axis('equal')
        theta = np.tanh((x_v1)/(y_v1))
        rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)], #rotation matrix
                              [np.sin(theta), np.cos(theta)]])
        transformed_mat = rotation_mat * coords

        # plot the transformed blob
        x_transformed, y_transformed = transformed_mat.A
        return x_transformed, y_transformed

    def compute_dfd(x1, y1, x2, y2):
        """Computes the discrete Frechet distance between two curves
        Parameters
        ----------
        x1 : np.ndarray
            An array containing x coordinates of first curve
        y1 : np.ndarray
            An array containing y coordinates of first curve
        x2 : np.ndarray
            An array containing x coordinates of second curve
        y2 : np.ndarray
            An array containing y coordinates of second curve
        Returns
        -------
        float
            The Frechet distance between the two curves
        """
        return similaritymeasures.frechet_dist((x1, y1), (x2, y2))

    def should_rotate(x, y):
        """Compares the Frechet distance between 1) untransformed data and (only) symmetricized data
        to 2) rotated data and rotated+symmetricized data. If 2 is smaller, then returns true.
        Parameters
        ----------
        x : np.ndarray
            An array containing x coordinates of fit datapoints
        y : np.ndarray
            An array contianing y coordinates of fit datapoints
        Returns
        -------
        bool
            True if the figure has a smaller Frechet distance when rotated then symmetricized; 
            false otherwise.
        """
        sym_x, sym_y = make_symmetric(x, y)
        dfd_1 = compute_dfd(x, y, sym_x, sym_y)

        rotated_x, rotated_y = rotate_figure(x, y)
        rotate_then_sym_x, rotate_then_sym_y = make_symmetric(rotated_x, rotated_y)
        dfd_2 = compute_dfd(rotated_x, rotated_y, rotate_then_sym_x, rotate_then_sym_y)

        return (dfd_1 > dfd_2)
