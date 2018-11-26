package iesl.cs.umass.edu.mapCG

import collection.mutable.{HashSet, ArrayBuffer}
import scala.Array

/**
 * Code used for 'MAP Inference in Chains Using Column Generation' NIPS 2012
 * Authors: David Belanger + Alexandre Passos
 */

object DelayedForwardBackward {
  def computeTransBounds(ds: Int, transMatrix: Int => (Int => Double)): (Array[Double], Array[Double], Array[Double], Array[Double]) = {
    val boundTransRow = Array.fill(ds)(Double.NegativeInfinity)
    val boundTransCol = Array.fill(ds)(Double.NegativeInfinity)
    val lowerBoundTransRow = Array.fill(ds)(Double.PositiveInfinity)
    val lowerBoundTransCol = Array.fill(ds)(Double.PositiveInfinity)

    // Compute row- and column-wise upper and lower bounds on the
    // entries of the transition matrix for more effective pruning
    for (i <- 0 until ds; j <- 0 until ds) {
      val t = transMatrix(i)(j)
      boundTransRow(i) = math.max(boundTransRow(i), t)
      boundTransCol(j) = math.max(boundTransCol(j), t)
      lowerBoundTransRow(i) = math.min(lowerBoundTransRow(i), t)
      lowerBoundTransCol(j) = math.min(lowerBoundTransCol(j), t)
    }

    val lowerBoundTrans = Array.ofDim[Double](ds)
    val upperBoundTrans = Array.ofDim[Double](ds)
    for(j<- 0 until ds){
      lowerBoundTrans(j) = lowerBoundTransCol(j) + lowerBoundTransRow(j)
      upperBoundTrans(j) = boundTransCol(j) + boundTransRow(j)
    }
    
    (boundTransRow, boundTransCol, lowerBoundTrans, upperBoundTrans)
  }
}


class DelayedForwardBackward(val transitionMatrix: Int => (Int => Double),
                             val localScores: Array[Array[Double]],
                             val boundTransRows: Array[Double], 
                             val boundTransColumns: Array[Double],
                             val lowerBoundTrans: Array[Double] = null: Array[Double], 
                             val upperBoundTrans: Array[Double] = null: Array[Double],
                             useCandidates: Boolean = true) {
  // The initial beam width
  var initialDomainSize = 1

  // The domains of all variables (which will grow)
  val domains = ArrayBuffer[ArrayBuffer[Int]]()

  // The size of the domains
  val ns = localScores(0).length
  val ds = ns

  // The number of variables
  val nv = localScores.length

  val fullDomain = (0 until ns).toSeq

  // The forward and backward dual variables
  val alphas = Array.fill[Double](nv, ns)(Double.NegativeInfinity)
  val betas = Array.fill[Double](nv, ns)(Double.NegativeInfinity)

  // The "argmaxes" which will be used in viterbi (forward and backward)
  val backPointers = Array.fill[Int](nv, ns)(-1)
  val betabackPointers = Array.fill[Int](nv, ns)(-1)

  val zeroArray = Array.fill[Double](ns)(0.0)

  // The earliest variable from which to start computing the forward messages
  var earliestUpdate = 0

  // The latest variable from which to start computing the backward messages
  var latestUpdate = nv

  // Whether the messages or domains changed at any position in the chain
  val changedAlphas = Array.fill[Boolean](nv)(true)
  val changedBetas = Array.fill[Boolean](nv)(true)
  val changedDomain = Array.fill[Boolean](nv)(true)

  // Whether each position in the chain has a domain of size one
  val domainOfSizeOne = Array.fill[Boolean](nv)(true)

  // The current maximum reduced cost at each position; used to
  // compute the dual objective
  val currentMaxReducedCosts = Array.ofDim[Double](nv)

  // The set of possible valid variables for each variable, computed a
  // priori from the local scores and bounds on the transitions
  val candidateDomains = Array.fill[ArrayBuffer[Int]](nv)(ArrayBuffer[Int]())

  // Whether some forward or backward message changed in the last iteration
  var changedSomeAlpha = true
  var changedSomeBeta = true

  var currentMAPScore = 0.0
  val currentMAPAssignments = Array.fill[Int](nv)(-1)

  // Whether to compute the current duality gap
  var computeDG = false

  // Main entry point
  def delayedSearch( maxCGIters: Int = 20, pdTerminate: Boolean = false, gapThresh: Double = .05): Seq[Int] = {
    // first, initialize domains
    computeDG = pdTerminate
    initializeDomains

    // next, do delayed search
    var stillLooking = true
    var cgIterations = 0
    earliestUpdate = 1
    latestUpdate = nv - 2
    var pdTerminated = false

    while (stillLooking && cgIterations < maxCGIters) {
      // first, do inference.
      if (cgIterations == 0) {
        // in first iter, do message passing everywhere
        delayedSearchWithArrays()
      } else{
         // inference in subsequent iterations is different because you
         // can avoid message passing in regions that weren't affected
          delayedSearchWithArrays2ndPass()
      }

      // then, update the domains (i.e. search for variables of positive reduced cost)
      val (s, dualObjective) = updatePairwiseDomains
      stillLooking = s

      // test for convergence
      if (!changedSomeAlpha || !changedSomeBeta) {
        stillLooking = false
      }

      //an optional alternative convergence criterion is if the duality gap is sufficiently small
      if (pdTerminate) {
        val po = getPrimalObjective()
        if (((dualObjective - po)/ po) < gapThresh) {
          stillLooking = false
          pdTerminated = true
        }
      }

      cgIterations += 1
    }

    // if it took too many iterations, we fall back to viterbi
    if (cgIterations == maxCGIters ) {
      initializeFullDomainsBackoff()
      delayedSearchWithArrays()
    }
    getCurrentMAP
    currentMAPAssignments
  }

  // throughtout, 'domains(i)' refers to the set of values for
  // variables(i) that we allow it to take on.  candidateDomains(i) is
  // a list of the settings that we consider adding to domains(i) in
  // our iterative algorithm
  def initializeDomains {
    for (i <- 0 until nv) {
      domains += ArrayBuffer[Int]()

      // we initialize the domain of every variable to be the setting
      // with the best local score
      val m = (0 until ns).maxBy(x => localScores(i)(x))
      domains(i) += m

      // now we initialize an array of candidate values to also
      // consider when doing column generation. This can be less than
      // the general domain for the variable. We prune some settings
      // by using the pruning strategy described at the end of section
      // 4.3 in our NIPS paper
      val bound = localScores(i)(m) + lowerBoundTrans(m)
      for (j <- 0 until ns) {
        if (localScores(i)(j) + upperBoundTrans(j) >= bound)
          candidateDomains(i) += j
      }
    }
  }

  // this is exactly the forward-backward viterbi algorithm, but where
  // variables' candidate domains are stored in 'candidateDomains'
  def delayedSearchWithArrays() {
    val ds = localScores(0).length

    // forward
    var vi = 0
    var i = 0
    while (i < ds) {
      alphas(vi)(i) = 0
      i += 1
    }

    vi = 1
    while (vi < nv) {
      val j = domains(vi - 1)(0)
      val s = alphas(vi - 1)(j) + localScores(vi - 1)(j)
      for (i <- candidateDomains(vi)) {
        alphas(vi)(i) = s + transitionMatrix(j)(i)
        backPointers(vi)(i) = j
      }
      vi += 1
    }

    // backward
    vi = nv - 1
    for (i <- fullDomain) {
      betas(vi)(i) = 0.0
    }
    vi -= 1

    while (vi >= 0) {
      val j = domains(vi + 1)(0)
      val s = localScores(vi + 1)(j) + betas(vi + 1)(j)
      for (i <- candidateDomains(vi)) {
        betas(vi)(i) = transitionMatrix(i)(j) + s
        betabackPointers(vi)(i) = j
      }

      vi -= 1
    }
  }

  // this does forward-backward, but intelligently given that it's a
  // second pass and only needs to pass messages where they could
  // potentially be different from the first pass
  def delayedSearchWithArrays2ndPass() {
    // forward
    var vi = 0
    val ub = math.max(earliestUpdate, 1)
    while (vi < ub) {
      changedAlphas(vi) = false
      vi += 1
    }
    var i = 0

    changedSomeAlpha = false
    var looking = true
    while (vi < nv) {
      changedAlphas(vi) = false
      if (changedDomain(vi - 1))
        looking = true
      if (looking) {
        // If the previous variable had a domain of size one we can
        // optimize
        if (domainOfSizeOne(vi) && vi > latestUpdate)
          looking = false
        if (!domainOfSizeOne(vi - 1)) {
          for (i <- candidateDomains(vi)) {
            val bp = backPointers(vi)(i)
            alphas(vi)(i) = Double.NegativeInfinity
            for (j <- domains(vi - 1)) {
              val s = alphas(vi - 1)(j) + localScores(vi - 1)(j) + transitionMatrix(j)(i)
              if (s > alphas(vi)(i)) {
                alphas(vi)(i) = s
                backPointers(vi)(i) = j
              }
            }
            changedAlphas(vi) = changedAlphas(vi) || (bp != backPointers(vi)(i))
          }
        }
      }
      changedSomeAlpha = changedSomeAlpha || changedAlphas(vi)
      vi += 1
    }

    // backward
    vi = nv - 1
    while (vi > latestUpdate) {
      changedBetas(vi) = false
      vi -= 1
    }

    i = 0
    changedBetas(vi) = false
    changedSomeBeta = false
    var betaTerminate = false
    vi -= 1
    while (vi >= 0) {
      changedBetas(vi) = false
      if (!betaTerminate) {
        if (domainOfSizeOne(vi) && (vi < earliestUpdate) )
          betaTerminate = true
        if (!domainOfSizeOne(vi + 1)) {
          for (i <- candidateDomains(vi)) {
            val bp = betabackPointers(vi)(i)
            betas(vi)(i) = Double.NegativeInfinity
            for (j <- domains(vi + 1)) {
              val s = transitionMatrix(i)(j) + localScores(vi + 1)(j) + betas(vi + 1)(j)
              if (s > betas(vi)(i)) {
                betas(vi)(i) = s
                betabackPointers(vi)(i) = j
              }
            }
            changedBetas(vi) = changedBetas(vi) || (bp != betabackPointers(vi)(i))
          }
        }
        changedSomeBeta = changedSomeBeta || changedBetas(vi)
      }
      vi -= 1
    }
  }


  // This function implements the reduced-cost oracle as described in the paper.
  def searchBoundPairwiseDomain(i: Int, j: Int): (Boolean, Double) = {
    var updatedSomething = false
    // the strategy here is to first bound the contribution from j and search all is which can conceivably
    // go above that bound; then, we bound the contribution from i and search all js which can conceivably
    // go above that bound; finally, we search only over the surviving (i,j) pairs for any actual candidate
    // the reduced cost is
    //   2*theta(xi)(xj) + lS(xi) + lS(xj) + alphas(j)(xj) + betas(i)(xi) - alphas(i)(xi) - betas(j)(xj)
    var boundJ = Double.NegativeInfinity
    for (xj <- candidateDomains(j)) {
      val s = boundTransColumns(xj) - alphas(j)(xj) + betas(j)(xj) + localScores(j)(xj)
      if (s > boundJ) boundJ = s
    }
    var candidateIs = ArrayBuffer[Int]()
    for (xi <- candidateDomains(i)) {
      if (-betas(i)(xi) + alphas(i)(xi) + localScores(i)(xi) + boundTransRows(xi) + boundJ > 0.0) candidateIs += xi
    }
    // now we'll bound i
    var boundI = Double.NegativeInfinity
    for (xi <- candidateIs) {
      val s = boundTransRows(xi) - betas(i)(xi) + alphas(i)(xi) + localScores(i)(xi)
      if (s > boundI) boundI = s
    }
    val candidateJs = ArrayBuffer[Int]()
    for (xj <- candidateDomains(j)) {
      if (-alphas(j)(xj) + boundTransColumns(xj) + betas(j)(xj) + localScores(j)(xj) + boundI > 0.0) candidateJs += xj
    }

    var max = Double.NegativeInfinity
    var nAboveZero = 0
    var maxi = -1
    var maxj = -1
    var first = true
    for (xi <- candidateIs; xj <- candidateJs; if ((!domains(i).contains(xi)) || (!domains(j).contains(xj)))) {
      val s = reducedCost(i, xi, xj)
      if (s > 0.0) {
        updatedSomething = true

        if (!domains(i).contains(xi)) {
          domains(i) += xi
          changedDomain(i) = true
          domainOfSizeOne(i) = false
        }

        if (!domains(j).contains(xj)) {
          domains(j) += xj
          changedDomain(j) = true
          domainOfSizeOne(j) = false
        }
        if (first) {
          if (i < earliestUpdate) earliestUpdate = i
          if (j > latestUpdate) latestUpdate = j
        }
        first = false
      }

      if (s > max) {
        max = s
      }
    }


    //do this because max reduced cost for instantiated edges is 0
    max = math.max(0, max)

    (updatedSomething, max)
  }

  def updatePairwiseDomains(): (Boolean, Double) = {
    earliestUpdate = math.max(1, nv - 1)
    latestUpdate = 0
    var updatedSomething = false
    var dualObjective = 0.0

    for (i <- 0 until (nv - 1)) {
      val j = i + 1
      changedDomain(i) = false
      if (j == nv - 1)
        changedDomain(j) = false

      // We only need to call the reduced cost oracle if the messages
      // have changed
      if (changedAlphas(j) || changedBetas(i)) {
        val (u, d) = searchBoundPairwiseDomain(i, j)
        dualObjective += d
        currentMaxReducedCosts(i) = d
        updatedSomething = updatedSomething | u
      } else {
        val d = currentMaxReducedCosts(i)
        dualObjective += d
      }
    }
    if(computeDG){
      val mAV = getBestAlphaIndex()._2
      val mBV = getBestBetaIndex()._2
      dualObjective += .5*(mAV + mBV)
    }

    (updatedSomething, dualObjective)
  }

  // Computes an array with the MAP assignments of all variables
  def getCurrentMAP: Double = {
    val (maxAlpha, maxi) = alphas(nv - 1).zipWithIndex.maxBy(v => v._1 + localScores(nv - 1)(v._2))
    val maxVal = maxAlpha + localScores(nv - 1)(maxi)
    var i = maxi
    for (n <- (0 until nv).reverse) {
      currentMAPAssignments(n) = i
      i = backPointers(n)(i)
    }
    currentMAPScore = maxVal
    maxVal
  }


  // This effectively backs off to viterbi by setting the domain of
  // each variable to be full.
  def initializeFullDomainsBackoff() {
    val nD = ns
    for (i <- 0 until nv) {
      domains(i).clear()
      for (j <- 0 until nD) {
        domains(i) += j
      }
    }
  }
  //see the paper for the derivation of the reduced cost expression
  def reducedCost(i: Int, xi: Int, xj: Int) = {
    val j = i + 1
    val sc = localScores(i)(xi) + localScores(j)(xj) + 2 * transitionMatrix(xi)(xj) + (betas(j)(xj) - betas(i)(xi)) + (alphas(i)(xi) - alphas(j)(xj))
    sc
  }

  def getBestAlphaIndex(): (Int,Double) = {
    var maxi = -1
    var maxVal = Double.NegativeInfinity
    for(i<- candidateDomains(nv - 1)){
        val s = alphas(nv - 1)(i) + localScores(nv - 1)(i)
        if(s >= maxVal){
          maxi = i
          maxVal = s
        }
     }
    (maxi,maxVal)
  }

  def getBestBetaIndex(): (Int,Double) = {
    var maxi = -1
    var maxVal = Double.NegativeInfinity
    for(i<- candidateDomains(0)){
        val s = betas(0)(i) + localScores(0)(i)
        if(s > maxVal){
          maxi = i
          maxVal = s
        }
     }
    (maxi,maxVal)
  }

  def getPrimalObjective(): Double = {
    var score = 0.0
    var i = getBestAlphaIndex()._1
    var in = -1
    for (n <- (0 until nv).reverse) {
      score += localScores(n)(i)
      currentMAPAssignments(n) = i
      in = backPointers(n)(i)
      if(n > 0)
        score += transitionMatrix(in)(i)
      i = in
    }
    score
  }
}
