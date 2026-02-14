"""
Data Loader Module
Handles loading and parsing CSV data files.
"""

import csv
from pathlib import Path


class DataLoader:
    """Loads and parses classmate data from CSV files."""

    def __init__(self, csv_path: Path, encoding: str = "cp1252"):
        """
        Initialize the DataLoader.

        Params:
            csv_path: Path to the CSV file
            encoding: File encoding (default: cp1252)
        """
        self.csv_path = csv_path
        self.encoding = encoding
        self.classmates_map: dict[str, str] = {}

    def load_data(self) -> dict[str, str]:
        """
        Load data from CSV file.

        Returns:
            Dictionary mapping paragraphs to names
        """
        with open(self.csv_path, newline="", encoding=self.encoding) as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quotechar='"')
            next(reader)  # Skip header

            for row in reader:
                name, paragraph = row
                self.classmates_map[paragraph] = name

        return self.classmates_map

    def get_paragraphs(self) -> list[str]:
        """
        Get list of paragraphs from loaded data.

        Returns:
            List of paragraph strings
        """
        return list(self.classmates_map.keys())

    def get_names(self) -> list[str]:
        """
        Get list of names from loaded data.

        Returns:
            List of name strings
        """
        return list(self.classmates_map.values())
