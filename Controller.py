__version__ = "0.0 alpha"

from tkinter.filedialog import askopenfilenames, askopenfilename, askdirectory
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Union
from tkinter import Tk
import numpy as np
import os

import BackgroundCorrection.algorithm as algorithm
import BackgroundCorrection.reader as reader
from BackgroundCorrection import jar
from BackgroundCorrection.units import convert_x
from BackgroundCorrection.util import apply_limits, normalize_area, normalize_max, ground, normalize_sum
from settings import load_settings

matplotlib.use("QtAgg")

READFILE_TYPES = [
    ("Raman Files", ".txt .raman"),
    ("Spectra Files", ".spc"),
    ("Chi Files", ".xy .dat"),
]


class DataSet:
    def __init__(self, files: List[reader.DataFile]):
        self.files: List[reader.DataFile] = files

        self.jar_file: Union[reader.DataFile, None] = None

        self.x_ranged = None
        self.range_selection = None

        self.result: reader.DataFile


class Controller:
    def __init__(self):
        # TODO: Load settings / set defaults
        self.settings = load_settings()
        self.bkg_params = {
            "fkt": algorithm.algorithm_by_index(self.settings.baseline.algorithm),
            "lam": self.settings.baseline.lam,
            "ratio": self.settings.baseline.ratio,
            "itermax": self.settings.baseline.itermax
        }

        self.files: List[reader.DataFile] = []

    def load_files(self):
        root = Tk()
        files = askopenfilenames(filetypes=READFILE_TYPES, title="Select files to process")
        root.destroy()

        if not files:
            files = []

            root = Tk()
            files_dir = askdirectory()
            root.destroy()

            files_subdirs = [subdir.path for subdir in os.scandir(files_dir)
                             if subdir.is_dir() and subdir.name != self.settings.io.out_dir]
            subdir_files = [[file.path for file in os.scandir(subdir) if os.path.isfile(file)]
                            for subdir in [files_dir, *files_subdirs]]

            for file_list in subdir_files:
                files.extend(file_list)

        # TODO: Log successful file selection

        dataFiles = reader.read_many(files, self.settings.io.head_row_count)
        self.files = dataFiles

    def extend_headers(self):
        params = {
            "alg": algorithm.algorithm_by_index(self.settings.baseline.algorithm).__name__,
            "itermax": self.settings.baseline.itermax,
            "lambda": self.settings.baseline.lam,
            "ratio": self.settings.baseline.ratio
        } if self.settings.baseline.enable else {}

        for file in self.files:
            file.extend_head(__version__, **params)

    def process_dataset(self, dataset: DataSet):
        print("Processing new dataset")

        # Check if x-Axes match through input files
        print("Checking x-Axes")
        xs = [file.x for file in dataset.files]
        assert (np.diff(np.vstack(xs).reshape(len(xs), -1),
                        axis=0) == 0).all(), "x-Axes through read files do not match"

        # Convert x-Axis between units and apply limits specified in settings
        x_unit_applied = np.vectorize(convert_x)(dataset.files[0].x, self.settings.data.unit_in,
                                                 self.settings.data.unit_out)
        dataset.x_ranged, dataset.range_selection = apply_limits(x_unit_applied, selection_range=(self.settings.data.range_start,
                                                                 self.settings.data.range_stop))

        # TODO: Option to use same jar for every dataset
        # Prepare jar-correction
        if self.settings.jar.enable:
            print("Loading reference file")
            root = Tk()
            filename = askopenfilename(filetypes=READFILE_TYPES, title="Select reference file")
            root.destroy()

            jar_file = jar.load_jar(filename, self.settings.jar.reference_file_head_row_count,
                                    (self.settings.jar.range_start, self.settings.jar.range_stop),
                                    dataset.range_selection)

        # Process data
        for file_index, file in enumerate(dataset.files):
            print(f"Processing file {file_index}")

            ys_ranged = []
            ys_jar_corrected = []
            ys_jar_scaled = []
            jar_scaling_factors = []
            ys_background_corrected = []
            ys_background_baseline = []
            ys_normalized = []
            ys_grounded = []
            ys_result = []

            file.x_ranged = dataset.x_ranged

            for column_index, intensity in enumerate(file.ys):
                print(f"Processing column {column_index} of file {file_index}")
                # Apply x-limits to intensity values
                intensity_ranged, _ = apply_limits(intensity, selection=dataset.range_selection)
                ys_ranged.append(intensity_ranged)

                # Apply jar-correction to intensity
                intensity_pre_bkg = intensity_ranged
                if self.settings.jar.enable:
                    intensity_jar_corrected, jar_intensity_scaled, jar_scaling_factor = jar.jar_correct(jar_file, intensity_ranged, **self.bkg_params)

                    ys_jar_corrected.append(intensity_jar_corrected)
                    ys_jar_scaled.append(jar_intensity_scaled)
                    jar_scaling_factors.append(jar_scaling_factor)

                    intensity_pre_bkg = intensity_jar_corrected

                # Perform background correction on prepared intensity data
                intensity_pre_ground = intensity_pre_bkg
                baseline = []
                if self.settings.baseline.enable:
                    intensity_corrected, baseline = algorithm.correct(intensity_pre_bkg, **self.bkg_params)
                    ys_background_corrected.append(intensity_corrected)
                    ys_background_baseline.append(baseline)

                    intensity_pre_ground = intensity_corrected

                # Ground processed intensity data
                intensity_pre_norm = intensity_pre_ground
                if self.settings.normalization.ground:
                    intensity_grounded = ground(intensity_pre_ground)
                    ys_grounded.append(intensity_grounded)

                    intensity_pre_norm = intensity_grounded

                # Normalize processed intensity data
                intensity_final = intensity_pre_norm
                if self.settings.normalization.area:
                    intensity_normalized = normalize_area(x=dataset.x_ranged, y=intensity_pre_norm)
                    ys_normalized.append(intensity_normalized)

                    intensity_final = intensity_normalized
                elif self.settings.normalization.sum:
                    intensity_normalized = normalize_sum(intensity_pre_norm)
                    ys_normalized.append(intensity_normalized)

                    intensity_final = intensity_normalized
                elif self.settings.normalization.max:
                    intensity_normalized = normalize_max(intensity_pre_norm)
                    ys_normalized.append(intensity_normalized)

                    intensity_final = intensity_normalized

                ys_result.append(intensity_final)

                # Plot data for first sample in first file
                if file_index == column_index == 0:
                    if self.settings.baseline.plot.enable and self.settings.baseline.enable:
                        if self.settings.baseline.plot.original:
                            plt.plot(dataset.x_ranged, intensity_ranged, label="Intensity (Original)")
                        if self.settings.baseline.plot.baseline:
                            plt.plot(dataset.x_ranged, baseline, label="Baseline")
                        if self.settings.baseline.plot.corrected:
                            plt.plot(dataset.x_ranged, intensity_pre_norm, label="Intensity (Corrected)")
                        if self.settings.baseline.plot.corrected_normalized:
                            plt.plot(dataset.x_ranged, intensity_final, label="Intensity (Corrected, Normalized")

                        plt.xlabel("x")
                        plt.ylabel("intensity")
                        plt.legend(loc="upper right")
                        plt.title(f"file {file_index}, column {column_index}")

                        plt.show()

                    if self.settings.jar.plot.enable and self.settings.jar.enable:
                        if self.settings.jar.plot.jar_original:
                            plt.plot(dataset.x_ranged, jar_file.ys[0], label="Jar Intensity (Original)")
                        if self.settings.jar.plot.jar_ranged:
                            plt.plot(jar_file.x_ranged, jar_file.ys_ranged[0], label="Jar Intensity (Ranged)")
                        if self.settings.jar.plot.jar_baseline:
                            plt.plot(jar_file.x_ranged, jar_file.ys_background_baseline[0], label="Jar Baseline (Ranged)")
                        if self.settings.jar.plot.jar_corrected:
                            plt.plot(jar_file.x_ranged, jar_file.ys_background_corrected[0], label="Jar Intensity (Corrected, Ranged)")
                        if self.settings.jar.plot.jar_scaled:
                            plt.plot(dataset.x_ranged, jar_intensity_scaled, label="Jar Intensity (Corrected, Scaled)")
                        if self.settings.jar.plot.intensity_original:
                            plt.plot(dataset.x_ranged, intensity_ranged, label="Intensity (Pre-Jar-Correction)")
                        if self.settings.jar.plot.intensity_corrected:
                            plt.plot(dataset.x_ranged, intensity_jar_corrected, label="Intensity (Jar-Corrected)")

                        plt.xlabel("x")
                        plt.ylabel("intensity")
                        plt.legend(loc="upper right")
                        plt.title(f"file {file_index}, column {column_index}")

                        plt.show()

            # Join column data per-file
            file.ys_ranged = np.array(ys_ranged)
            file.ys_jar_corrected = np.array(ys_jar_corrected)
            file.jar_scaling_factors = np.array(jar_scaling_factors)
            file.ys_background_corrected = np.array(ys_background_corrected)
            file.ys_background_baseline = np.array(ys_background_baseline)
            file.ys_grounded = np.array(ys_grounded)
            file.ys_normalized = np.array(ys_normalized)

            file.ys_result = np.array(ys_result)
            file.x_result = dataset.x_ranged

        # Join results of individual files into single array
        dataset.x_result = dataset.files[0].x_result if len(dataset.files) != 0 else None
        dataset.ys_result = np.vstack([file.ys_result for file in dataset.files])

        # TODO: Get ROI integration data on whole DataSet intensity

        # TODO: Construct "Output" DataFile instance with result data
        # TODO: Write output data to file

        # TODO: Plot data

    def run(self):
        # TODO: Current approach expects same x for all files -> Group files in datasets?

        # Load files and extend headers with parameters specified in settings
        self.load_files()
        self.extend_headers()

        dataset = DataSet(self.files)

        self.process_dataset(dataset)


if __name__ == "__main__":
    controller = Controller()
    controller.run()

