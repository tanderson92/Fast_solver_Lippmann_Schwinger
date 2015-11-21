# script to test the quadratures


y = 0.3
(x,w) = gaussLegendreQuad(40, -1., 1.)
N = length(x[:])
M = evalLegendrePol(N,x)
w1 = modifiedGLWeights1(w[:], x[:], y)
# checking versus principal value computed
# in Mathematica
@assert abs(-1.814288237478133 - sum(M[:,2].*w1))< 1e-14

# checking the integration against the log
 w2 = modifiedGLWeights2(w[:], x[:], y);
 @assert abs(-0.021523324006526935- sum(M[:,10].*w2))< 1e-13
