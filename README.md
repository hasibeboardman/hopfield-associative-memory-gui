# Hopfield Neural Network Pattern Recognition (GUI)

This project implements a Hopfield Neural Network using Python and Tkinter, allowing users to interactively draw, train, and test binary image patterns. The network learns binary patterns and can recall similar patterns using pattern completion and associative memory capabilities.

## Features
- Graphical User Interface (GUI) with Tkinter
- Learn up to 3 binary patterns drawn on a 10x10 grid
- Query the network with a noisy pattern
- Visualize the recalled pattern based on the closest learned pattern
- Uses Hamming distance for similarity checking

## Technologies Used
- Python 3
- NumPy
- Tkinter

## How It Works
1. Draw a pattern on any of the 3 "Learned Pattern" canvases.
2. Click “Learn” to store the pattern in the Hopfield Network.
3. Draw a partial/noisy pattern in the “Query Pattern” canvas.
4. Click “Query” to recall the closest learned pattern.
5. View the result in the “Result Pattern” canvas.

## How to Run
Make sure you have Python 3 installed.

```bash
python hopfield_pattern_recall.py
