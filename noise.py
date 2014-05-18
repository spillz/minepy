#
# A speed-improved simplex noise algorithm for 2D, 3D and 4D in Java.
#
# Based on example code by Stefan Gustavson (stegu@itn.liu.se).
# Optimisations by Peter Eastman (peastman@drizzle.stanford.edu).
# Better rank ordering method by Stefan Gustavson in 2012.
#
# This could be speeded up even further, but it's useful as it is.
#
# Version 2012-03-09
#
# This code was placed in the public domain by its original author,
# Stefan Gustavson. You may use it as you see fit, but
# attribution is appreciated.
#
#
import numpy
import itertools


grad3 = numpy.array([(1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0),
                             (1,0,1),(-1,0,1),(1,0,-1),(-1,0,-1),
                             (0,1,1),(0,-1,1),(0,1,-1),(0,-1,-1)])

grad4 = numpy.array([(0,1,1,1),(0,1,1,-1),(0,1,-1,1),(0,1,-1,-1),
               (0,-1,1,1),(0,-1,1,-1),(0,-1,-1,1),(0,-1,-1,-1),
               (1,0,1,1),(1,0,1,-1),(1,0,-1,1),(1,0,-1,-1),
               (-1,0,1,1),(-1,0,1,-1),(-1,0,-1,1),(-1,0,-1,-1),
               (1,1,0,1),(1,1,0,-1),(1,-1,0,1),(1,-1,0,-1),
               (-1,1,0,1),(-1,1,0,-1),(-1,-1,0,1),(-1,-1,0,-1),
               (1,1,1,0),(1,1,-1,0),(1,-1,1,0),(1,-1,-1,0),
               (-1,1,1,0),(-1,1,-1,0),(-1,-1,1,0),(-1,-1,-1,0)])

p = numpy.array( [151,160,137,91,90,15,
    131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
    190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
    88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
    77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
    102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
    135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
    5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
    223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
    129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
    251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
    49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180] )
#p = numpy.random.randint(256,size=256)
# To remove the need for index wrapping, double the permutation table length
perm = numpy.arange(512,dtype='i2')
perm = p[perm & 255]

permMod12 = perm % 12

#Skewing and unskewing factors for 2, 3, and 4 dimensions
F2 = 0.5*(3.0**0.5-1.0)
G2 = (3.0-3.0**0.5)/6.0
F3 = 1.0/3.0
G3 = 1.0/6.0
F4 = (5.0**0.5-1.0)/4.0
G4 = (5.0-5.0**0.5)/20.0


# returns floor of floating point array by coercing to integer
def fastfloor(x):
    return numpy.array(numpy.floor(x), dtype = numpy.int32)

def dot(g, *v):
    s = 0
    for i in range(len(v)):
        s = s + g[i]*v[i]
    return s
        
# 2D simplex noise
def noise2(xin, yin): 
    # Skew the input space to determine which simplex cell we're in
    s = (xin+yin)*F2 # Hairy factor for 2D
    i = fastfloor(xin+s)
    j = fastfloor(yin+s)
    t = (i+j)*G2
    X0 = i-t # Unskew the cell origin back to (x,y) space
    Y0 = j-t
    x0 = xin-X0 # The x,y distances from the cell origin
    y0 = yin-Y0
    # For the 2D case, the simplex shape is an equilateral triangle.
    # Determine which simplex we are in.
    i1 = x0>y0  # lower triangle, XY order: (0,0)->(1,0)->(1,1)
    j1 = 1 - i1 # upper triangle, YX order: (0,0)->(0,1)->(1,1)
    # A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
    # a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
    # c = (3-sqrt(3))/6
    x1 = x0 - i1 + G2 # Offsets for middle corner in (x,y) unskewed coords
    y1 = y0 - j1 + G2
    x2 = x0 - 1.0 + 2.0 * G2 # Offsets for last corner in (x,y) unskewed coords
    y2 = y0 - 1.0 + 2.0 * G2
    
    # Work out the hashed gradient indices of the three simplex corners
    ii = i & 255
    jj = j & 255
    gi0 = permMod12[ii+perm[jj]]
    gi1 = permMod12[ii+i1+perm[jj+j1]]
    gi2 = permMod12[ii+1+perm[jj+1]]
    # Calculate the contribution from the three corners
    t0 = 0.5 - x0*x0-y0*y0
    tn = t0<0
    t0 =  tn*t0 + (1-tn)*t0*t0
    n0 = (1-tn) * t0 * t0 * dot(grad3[gi0].T, x0, y0) ##FIXME

    t1 = 0.5 - x1*x1-y1*y1
    tn = t1<0
    t1 = tn * t1 + (1-tn)*t1*t1
    n1 = (1-tn) * t1 * t1 * dot(grad3[gi1].T, x1, y1)

    t2 = 0.5 - x2*x2-y2*y2
    tn = t2<0
    t2 = tn * t2 + (1-tn)*t2*t2
    n2 = (1-tn) * t2 * t2 * dot(grad3[gi2].T, x2, y2)

    # Add contributions from each corner to get the final noise value.
    # The result is scaled to return values in the interval [-1,1].
    return 70.0 * (n0 + n1 + n2)



# 3D simplex noise
def noise3(xin, yin, zin): 
    # Skew the input space to determine which simplex cell we're in
    s = (xin+yin+zin)*F3 # Very nice and simple skew factor for 3D
    i = fastfloor(xin+s)
    j = fastfloor(yin+s)
    k = fastfloor(zin+s)
    t = (i+j+k)*G3
    X0 = i-t # Unskew the cell origin back to (x,y,z) space
    Y0 = j-t
    Z0 = k-t
    x0 = xin-X0 # The x,y,z distances from the cell origin
    y0 = yin-Y0
    z0 = zin-Z0
    # For the 3D case, the simplex shape is a slightly irregular tetrahedron.
    # Determine which simplex we are in.
    xy = x0>=y0
    yz = y0>=z0
    xz = x0>=z0
    i1 = xy&(yz|xz)
    i2 = xy | ~xy&yz&xz
    j1 = ~xy&yz
    j2 = xy&yz | ~xy
    k1 = xy&~yz&~xz | ~xy&~yz
    k2 = xy&~yz | ~xy&(~yz|~xz)
    
##        if(x0>=y0) {
##          if(y0>=z0)
##            { i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; } # X Y Z order
##            else if(x0>=z0) { i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; } # X Z Y order
##            else { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; } # Z X Y order
##          }
##        else { # x0<y0
##          if(y0<z0) { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; } # Z Y X order
##          else if(x0<z0) { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; } # Y Z X order
##          else { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; } # Y X Z order
##        }
    # A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
    # a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
    # a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
    # c = 1/6.
    x1 = x0 - i1 + G3 # Offsets for second corner in (x,y,z) coords
    y1 = y0 - j1 + G3
    z1 = z0 - k1 + G3
    x2 = x0 - i2 + 2.0*G3 # Offsets for third corner in (x,y,z) coords
    y2 = y0 - j2 + 2.0*G3
    z2 = z0 - k2 + 2.0*G3
    x3 = x0 - 1.0 + 3.0*G3 # Offsets for last corner in (x,y,z) coords
    y3 = y0 - 1.0 + 3.0*G3
    z3 = z0 - 1.0 + 3.0*G3
    # Work out the hashed gradient indices of the four simplex corners
    ii = i & 255
    jj = j & 255
    kk = k & 255
    gi0 = permMod12[ii+perm[jj+perm[kk]]]
    gi1 = permMod12[ii+i1+perm[jj+j1+perm[kk+k1]]]
    gi2 = permMod12[ii+i2+perm[jj+j2+perm[kk+k2]]]
    gi3 = permMod12[ii+1+perm[jj+1+perm[kk+1]]]
    # Calculate the contribution from the four corners
    t0 = 0.5 - x0*x0 - y0*y0 - z0*z0
    tn = t0<0
    t0 = tn*t0 + (1-tn)*t0*t0
    n0 = (1-tn)*t0 * t0 * dot(grad3[gi0].T, x0, y0, z0)
    
    t1 = 0.5 - x1*x1 - y1*y1 - z1*z1
    tn = t1<0
    t1 = tn*t1 + (1-tn)*t1*t1
    n1 = (1-tn)*t1 * t1 * dot(grad3[gi1].T, x1, y1, z1)
    
    t2 = 0.5 - x2*x2 - y2*y2 - z2*z2
    tn = t2<0
    t2 = tn*t2 + (1-tn)*t2*t2
    n2 = (1-tn)*t2 * t2 * dot(grad3[gi2].T, x2, y2, z2)
    
    t3 = 0.5 - x3*x3 - y3*y3 - z3*z3
    tn = t3<0
    t3 = tn*t3 + (1-tn)*t3*t3
    n3 = (1-tn)*t3 * t3 * dot(grad3[gi3].T, x3, y3, z3)
    
    # Add contributions from each corner to get the final noise value.
    # The result is scaled to stay just inside [-1,1]
    return 32.0*(n0 + n1 + n2 + n3)
        
class SimplexNoise2:
    '''
    unused, eventually replace SimplexNoise with a more sane
    and readable version using sine waves
    '''
    def __init__(self,seed=None):
        if seed:
            numpy.random.seed(seed)
            p = numpy.random.randint(256,size=256)
            # To remove the need for index wrapping, double the permutation table length
            perm0 = numpy.arange(512,dtype='i2')
            self.perm0 = p[perm0 & 255]
        else:
            self.perm0 = perm
    
    def noise(self, Z):
        N = Z.shape[-1] #number of dimensions
        N1 = N+1 # number of simplices
        

#  # N-D simplex noise, better simplex rank ordering method 2012-03-09
class SimplexNoise:
    def __init__(self,seed=None):
        if seed:
            numpy.random.seed(seed)
            p = numpy.random.randint(256,size=256)
            # To remove the need for index wrapping, double the permutation table length
            perm0 = numpy.arange(512,dtype='i2')
            self.perm0 = p[perm0 & 255]
        else:
            self.perm0 = perm

    def noise(self, Z):
        # Skew the (x,y,z,w) space to determine which cell of 24 simplices we're in
        N = Z.shape[-1] #number of dimensions
        N1 = N+1 # number of simplices
        Fn = 1.0*(N1**0.5 - 1)/N
        Gn = 1.0*(N1 - N1**0.5)/N/N1

        #skew the Z data and store in z0
        s = Z.sum(-1) * Fn # Factor for skewing
        i = fastfloor(Z+s[:,numpy.newaxis]) #nearest value
        t = (i.sum(-1) * Gn) # Factor for unskewing
        Z0 = i - t[:,numpy.newaxis]
        z0 = Z - Z0

        # Use magnitude ordering to determine the simplices that the point z0 is located in
        rank = numpy.zeros(Z.shape)
        for l,k in itertools.combinations(range(N),2):
            rank[:,k] += z0[:,k]>=z0[:,l]
            rank[:,l] += z0[:,k]<z0[:,l]

        # ind will contain the skewed indices of the N+1 simplices
        b = numpy.arange(N+1)[:,numpy.newaxis,numpy.newaxis]
        ind = rank>= N - b
        # zk contains the skewed locations of the N+1 simplices
        zk = z0 - ind + 1.0 * b * Gn
        
        indi = (ind) % 255 + i
        # the gradients are randomly assigned to each simplex
        grad = ((0,-1,1),)*N
        grad = numpy.array(list(itertools.product(*grad))[1:])
        grad = grad[numpy.abs(grad).sum(-1)>=N-1]

        gik = 0
        for x in range(N-1,-1,-1):
            gik = self.perm0[indi[:,:,x] + gik]
        gik = gik%(grad.shape[0])#(2**(N+1))
        # Calculate the contribution from the simplices
        tk = 0.5 - (zk*zk).sum(-1)
        tp = tk>=0
        tk = tp * tk * tk
        nk = tp * tk * tk * (grad[gik]*zk).sum(-1)

        # Sum up and scale the result to cover the range [-1,1]
        return nk.sum(0) * (2**6 )#(2**(N+1)-N-1)



def noisen(Z, seed=None):
    s = SimplexNoise(seed)
    return s.noise(Z)


if __name__ == '__main__':
    import time
    from PIL import Image, ImageSequence

    l = []
    l2 = []
    n2 = []
    t=time.time()
    arr2 = numpy.mgrid[0:8:0.1,0:8:0.1].T
    print 'arr2 mgrid',time.time()-t
    shape2 = arr2.shape
    arr2 = arr2.reshape((shape2[0]*shape2[1],2))
    t=time.time()
    arr3 = numpy.mgrid[0:8:0.1,0:8:0.1,0:8:0.1].T
    shape3 = arr3.shape
    arr3 = arr3.reshape((shape3[0]*shape3[1]*shape3[2],3))
    print 'arr3 mgrid',time.time()-t

    print 'gen noise'
    #n = numpy.array(n2).reshape(80,80)
    t=time.time()
    n = noisen(arr2,seed=3332).reshape(shape2[0],shape2[1])
    print 'arr2 noise',time.time()-t
    t=time.time()
    n3 = noisen(arr3)
    print 'arr3 noise',time.time()-t
    print 'STATS'
    print '######'
    print n.min(),n.max(),numpy.average(n)
    print n3.min(),n3.max(),numpy.average(n3)
    n = numpy.array((n - n.min()) / (n.max()-n.min())*255,dtype='u1')
    im = Image.fromarray(n,'L')
    print im.size
    im.save('noise2.png')
    
