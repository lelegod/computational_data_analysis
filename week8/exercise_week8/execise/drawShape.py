def drawShape(mu, conlist):
   # Matlabs drawshape is a strange un-intuative function, here's how you do it
   for i in range(np.size(conlist, axis = 0)): # How many different lines exist in the data (7)
       xpoints = mu[conlist[i,0]:conlist[i,1]+1]
       ypoints = mu[conlist[i,0] +  58:conlist[i,1] + 59]
       
       if conlist[i][2] == 1: # If it is a closed loop
           xpoints = np.append(xpoints, xpoints[0])
           ypoints = np.append(ypoints, ypoints[0])
       plt.plot(xpoints, ypoints, color = "b")
   plt.title('Mean face')
   plt.axis('equal')
   plt.show()
