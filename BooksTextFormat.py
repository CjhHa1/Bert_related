import glob
import os

class BookscorpusTextFormatting:
    def __init__(self, books_path, output_filename, recursive = False):
        self.books_path = books_path
        self.recursive = recursive
        self.output_filename = output_filename


    # This puts one book per line
    def merge(self):
        with open(self.output_filename, mode='w', newline='\n',encoding="utf-8") as ofile:
            for filename in glob.glob(self.books_path + '/' + '*.txt', recursive=True):
                print(filename)
                with open(filename, mode='r', encoding='utf-8-sig', newline='\n') as file:
                    for line in file:
                        if line.strip() != '':
                            ofile.write(line.strip() + ' ')
                ofile.write("\n\n")