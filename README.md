# minepy

A fork of fogleman's simple Minecraft-inspired demo written in Python.

https://github.com/fogleman/minecraft


## Goals and Vision

To be a simple engine for building minecraft like games using python

## Technical

Uses pyglet and numpy.

## How to Run

    pip install pyglet
    pip install numpy
    git clone https://github.com/spillz/minepy.git
    cd minepy
    python main.py

### Mac

On Mac OS X, you may have an issue with running Pyglet in 64-bit mode. Try running Python in 32-bit mode first:

    arch -i386 python main.py

If that doesn't work, set Python to run in 32-bit mode by default:

    defaults write com.apple.versioner.python Prefer-32-Bit -bool yes 

This assumes you are using the OS X default Python.  Works on Lion 10.7 with the default Python 2.7, and may work on other versions too.  Please raise an issue if not.
    
Or try Pyglet 1.2 alpha, which supports 64-bit mode:  

    pip install https://pyglet.googlecode.com/files/pyglet-1.2alpha1.tar.gz 

### If you don't have pip or git

For pip:

- Mac or Linux: install with `sudo easy_install pip` (Mac or Linux) - or (Linux) find a package called something like 'python-pip' in your package manager.
- Windows: [install Distribute then Pip](http://stackoverflow.com/a/12476379/992887) using the linked .MSI installers.

For git:

- Mac: install [Homebrew](http://mxcl.github.com/homebrew/) first, then `brew install git`.
- Windows or Linux: see [Installing Git](http://git-scm.com/book/en/Getting-Started-Installing-Git) from the _Pro Git_ book.

See the [wiki](https://github.com/fogleman/Minecraft/wiki) for this project to install Python, and other tips.

## How to Play

### Moving

- W: forward
- S: back
- A: strafe left
- D: strafe right
- Mouse: look around
- Space: jump
- Tab: toggle flying mode

### Building

# Use the number keys to select the type of block to create:
    - 1: dirt with grass
    - 2: grass
    - 3: sand
    - etc
- Mouse left-click: remove block
- Mouse right-click: create block

### Quitting

- ESC: release mouse, then close window

# Licenses

#Code - GPLv3

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Texture Pack - Faithful Venom v1.5

	Faith Venom is licensed CC BY-NC-SA 3.0
	http://minecraft.curseforge.com/texture-packs/51244-faithfulvenom-32x-32x
