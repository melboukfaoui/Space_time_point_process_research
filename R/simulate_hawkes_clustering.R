simulate_hawkes_clustering <- function(tmax, W, mu, K, beta, sigma) {
  # simulate initial parents
  xmin <- W[1]; xmax <- W[2]; ymin <- W[3]; ymax <- W[4]
  total_n <- (xmax-xmin)*(ymax-ymin)*tmax*mu
  homog_p <- rpois(1, lambda = total_n)
  print(paste("simulating ", homog_p, " initial parents\n"))
  xs <- runif(homog_p, min=xmin, max=xmax)
  ys <- runif(homog_p, min=ymin, max=ymax)
  ts <- runif(homog_p, min=0, max=tmax)
  
  # simulate offspring
  off_n <- rpois(homog_p, K)
  parent_xs <- xs[off_n > 0]; parent_ys <- ys[off_n > 0]; parent_ts <- ts[off_n > 0]
  off_n <- off_n[off_n > 0]
  
  while(length(parent_xs) > 0) {
    print(paste("simulating ", sum(off_n), " children\n"))
    allchild_xs <- allchild_ys <- allchild_ts <- c()  
    
    # simulate the children points 
    for(i in seq_along(parent_ts)) {
      child_ts <- parent_ts[i]+rexp(off_n[i], rate = beta)
      child_xs <- parent_xs[i]+rnorm(off_n[i], mean=0, sd=sigma)
      child_ys <- parent_ys[i]+rnorm(off_n[i], mean=0, sd=sigma)
      allchild_xs <- c(allchild_xs, child_xs) 
      allchild_ys <- c(allchild_ys, child_ys) 
      allchild_ts <- c(allchild_ts, child_ts)
    }
    
    xs <- c(xs, allchild_xs); ys <- c(ys, allchild_ys); ts <- c(ts, allchild_ts)
    off_n <- rpois(length(allchild_ts), K)
    parent_xs <- allchild_xs[off_n > 0]; parent_ys <- allchild_ys[off_n > 0]; parent_ts <- allchild_ts[off_n > 0]
    off_n <- off_n[off_n>0]
  }
  return(data.frame(x=xs,y=ys,t=ts))
}

simulate_hawkes_clustering_temporal <- function(tmax, W, mu, K, beta, sigma) {
  library("poweRlaw")
  # simulate initial parents
  xmin <- W[1]; xmax <- W[2]; ymin <- W[3]; ymax <- W[4]
  total_n <- (xmax-xmin)*(ymax-ymin)*tmax*mu
  homog_p <- rpois(1, lambda = total_n)
  print(paste("simulating ", homog_p, " initial parents\n"))
  xs <- runif(homog_p, min=xmin, max=xmax)
  ys <- runif(homog_p, min=ymin, max=ymax)
  ts <- runif(homog_p, min=0, max=tmax)
  
  # simulate offspring
  off_n <- rpois(homog_p, K)
  parent_xs <- xs[off_n > 0]; parent_ys <- ys[off_n > 0]; parent_ts <- ts[off_n > 0]
  off_n <- off_n[off_n > 0]
  
  while(length(parent_xs) > 0) {
    print(paste("simulating ", sum(off_n), " children\n"))
    allchild_xs <- allchild_ys <- allchild_ts <- c()  
    
    # simulate the children points 
    for(i in seq_along(parent_ts)) {
      child_ts <- parent_ts[i]+rexp(off_n[i], rate = beta)
      child_xs <- parent_xs[i]+rnorm(off_n[i], mean=0, sd=sigma)
      child_ys <- parent_ys[i]+rnorm(off_n[i], mean=0, sd=sigma)
      allchild_xs <- c(allchild_xs, child_xs) 
      allchild_ys <- c(allchild_ys, child_ys) 
      allchild_ts <- c(allchild_ts, child_ts)
    }
    
    xs <- c(xs, allchild_xs); ys <- c(ys, allchild_ys); ts <- c(ts, allchild_ts)
    off_n <- rpois(length(allchild_ts), K)
    parent_xs <- allchild_xs[off_n > 0]; parent_ys <- allchild_ys[off_n > 0]; parent_ts <- allchild_ts[off_n > 0]
    off_n <- off_n[off_n>0]
  }
  return(data.frame(x=xs,y=ys,t=ts))
}

simulate_hawkes_clustering_spatial <- function(tmax, W, mu, K, beta, sigma) {
  # simulate initial parents
  xmin <- W[1]; xmax <- W[2]; ymin <- W[3]; ymax <- W[4]
  total_n <- (xmax-xmin)*(ymax-ymin)*tmax*mu
  homog_p <- rpois(1, lambda = total_n)
  print(paste("simulating ", homog_p, " initial parents\n"))
  xs <- runif(homog_p, min=xmin, max=xmax)
  ys <- runif(homog_p, min=ymin, max=ymax)
  ts <- runif(homog_p, min=0, max=tmax)
  
  # simulate offspring
  off_n <- rpois(homog_p, K)
  parent_xs <- xs[off_n > 0]; parent_ys <- ys[off_n > 0]; parent_ts <- ts[off_n > 0]
  off_n <- off_n[off_n > 0]
  
  while(length(parent_xs) > 0) {
    print(paste("simulating ", sum(off_n), " children\n"))
    allchild_xs <- allchild_ys <- allchild_ts <- c()  
    
    # simulate the children points 
    for(i in seq_along(parent_ts)) {
      child_ts <- parent_ts[i]+rexp(off_n[i], rate = beta)
      child_xs <- parent_xs[i]+rnorm(off_n[i], mean=0, sd=sigma)
      child_ys <- parent_ys[i]+rnorm(off_n[i], mean=0, sd=sigma)
      allchild_xs <- c(allchild_xs, child_xs) 
      allchild_ys <- c(allchild_ys, child_ys) 
      allchild_ts <- c(allchild_ts, child_ts)
    }
    
    xs <- c(xs, allchild_xs); ys <- c(ys, allchild_ys); ts <- c(ts, allchild_ts)
    off_n <- rpois(length(allchild_ts), K)
    parent_xs <- allchild_xs[off_n > 0]; parent_ys <- allchild_ys[off_n > 0]; parent_ts <- allchild_ts[off_n > 0]
    off_n <- off_n[off_n>0]
  }
  return(data.frame(x=xs,y=ys,t=ts))
}

simulate_hawkes_clustering_bg <- function(tmax, W, mu, K, beta, sigma) {
  # simulate initial parents
  xmin <- W[1]; xmax <- W[2]; ymin <- W[3]; ymax <- W[4]
  total_n <- (xmax-xmin)*(ymax-ymin)*tmax*mu
  homog_p <- rpois(1, lambda = total_n)
  print(paste("simulating ", homog_p, " initial parents\n"))
  xs <- runif(homog_p, min=xmin, max=xmax)
  ys <- runif(homog_p, min=ymin, max=ymax)
  ts <- runif(homog_p, min=0, max=tmax)
  
  # simulate offspring
  off_n <- rpois(homog_p, K)
  parent_xs <- xs[off_n > 0]; parent_ys <- ys[off_n > 0]; parent_ts <- ts[off_n > 0]
  off_n <- off_n[off_n > 0]
  
  while(length(parent_xs) > 0) {
    print(paste("simulating ", sum(off_n), " children\n"))
    allchild_xs <- allchild_ys <- allchild_ts <- c()  
    
    # simulate the children points 
    for(i in seq_along(parent_ts)) {
      child_ts <- parent_ts[i]+rexp(off_n[i], rate = beta)
      child_xs <- parent_xs[i]+rnorm(off_n[i], mean=0, sd=sigma)
      child_ys <- parent_ys[i]+rnorm(off_n[i], mean=0, sd=sigma)
      allchild_xs <- c(allchild_xs, child_xs) 
      allchild_ys <- c(allchild_ys, child_ys) 
      allchild_ts <- c(allchild_ts, child_ts)
    }
    
    xs <- c(xs, allchild_xs); ys <- c(ys, allchild_ys); ts <- c(ts, allchild_ts)
    off_n <- rpois(length(allchild_ts), K)
    parent_xs <- allchild_xs[off_n > 0]; parent_ys <- allchild_ys[off_n > 0]; parent_ts <- allchild_ts[off_n > 0]
    off_n <- off_n[off_n>0]
  }
  return(data.frame(x=xs,y=ys,t=ts))
}