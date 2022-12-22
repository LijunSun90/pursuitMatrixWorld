ffmpeg -r 30 -i frames/MatrixWorld%4d.png -c:v libx264 -vf fps=24 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" pursuit.mp4
