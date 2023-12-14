########Second Stage###################
##Prepare for outcome
##Sampling Outcome Stage
#Basis in Mediation Data

l=13;K=6;Grid.num=200;bandwidth=2;simple=1;simu_time=20000

construct_B<-function(x,knots)
{
  return(cbind(1, x, matrix(unlist(lapply(
    x,
    FUN = function(x) {
      exp(-bandwidth*(x - knots) ^ 2)
    }
  )), nrow = length(x), byrow = T)))
}

time_grids_obs=unique(Y_data$Time)
knots_y=quantile(time_grids_obs,seq(0,1,length=l+2))[-c(1,l+2)]

#Recalculate the time matrix
B_list=aggregate(Y_data$Time,by=list(Y_data$ID),FUN=function(x){construct_B(x,knots_y)})
YB_matrix=B_list$x

Time_Grid=seq(0,1,length=Grid.num)

#Construct Time Grid B
YB_Grid=construct_B(Time_Grid,knots_y)
YB_Grid_Product=t(YB_Grid)%*%YB_Grid


n_unit=length(unique(Y_data$ID))
n_total=length(Y_data$Time)

X_Y=data.frame(rep(1.0,n_total))

id_index=aggregate(1:nrow(X_Y),by=list(Y_data$ID),FUN=function(x){x})
id_index=id_index$x


#Outcome
Obs=Y_data$Outcome
Obs_list=aggregate(Y_data$Outcome, by=list(Y_data$ID), function(x){x})
Obs_value=Obs_list$x

#Treatment Indicator
Treatment_Ind=aggregate(Y_data$Treatment,by=list(Y_data$ID),FUN=function(x){mean(x)})$x
N_0=sum(Treatment_Ind==0)
N_1=n_unit-N_0

#Penalty Function
Omega=diag(c(0,0,rep(1,l)))


#MCMC Time
#Parameter for Covariates
theta=matrix(0.0,ncol=ncol(X_Y)+1,nrow=simu_time)
#Principal Score
Beta_ini=matrix(runif(K*n_unit,-0.5,0.5),ncol=K,nrow=n_unit)
Beta=rep(list(NA),simu_time)
Beta[[1]]=Beta_ini

#Precision Parameter
lambda=matrix(2,ncol=K,nrow=simu_time)
Psi_ini=matrix(runif(K*(l+2),-1,1),ncol=K,nrow=l+2)
Psi=rep(list(NA),simu_time)
Psi[[1]]=Psi_ini

#Precison Parameters
sigma_noise=rep(0,simu_time);sigma_noise[1]=1
sigma_K=matrix(0,ncol=K,nrow=simu_time);sigma_K[1,]=K:1

sigma_theta=10^2

#TE Parameter
tau_0=tau_1=matrix(0,ncol=K,nrow=simu_time)

#Parameter for MGP
#For causal parameters
tau_delta=matrix(0,ncol=K,nrow=simu_time);tau_delta[1,]=rep(1,K)
#For RD
rd_delta=matrix(0,ncol=K,nrow=simu_time);rd_delta[1,]=rep(1,K)
sigma_tau=100
#Hyper
#amu=ard=matrix(0,ncol=2,nrow=simu_time);amu[1,]=c(1,2);ard[1,]=c(1,2)
ard=matrix(0,ncol=2,nrow=simu_time);ard[1,]=c(1,2)

phi_ini=matrix(1,ncol=K,nrow=n_unit)
phi=rep(list(NA),simu_time)
phi[[1]]=phi_ini

#Step Size for MH
step_size=1
v=1

AUG_X=as.matrix(cbind(X_Y,Y_data$Mediator))
##Gibbs Sampler
for (t in 2:simu_time){
  #Fix k
  ####################################
  #Sample Eigen function
  ####################################
  Psi[[t]]=Psi[[t-1]]
  Beta[[t]]=Beta[[t-1]]
  for (k in sample(1:K,K)){
    Q_k=Reduce('+',lapply(
      1:n_unit,
      FUN = function(x) {
        t(YB_matrix[[x]]) %*% YB_matrix[[x]] * Beta[[t]][x, k]^2
      }
    ))/sigma_noise[t-1]+lambda[t-1,k]*Omega


    l_k=Reduce('+',lapply(1:n_unit,FUN=function(i){t(YB_matrix[[i]])%*%(Obs_value[[i]]-AUG_X[id_index[[i]],]%*%theta[t-1,]-
                                                                          YB_matrix[[i]]%*%Psi[[t]][,-k]%*%Beta[[t]][i,-k])*Beta[[t]][i,k]}))/sigma_noise[t-1]
    #Add a constraint
    C_k=t(cbind(c(1,rep(0,l+1)),Psi[[t]][,-k]))%*%YB_Grid_Product

    Q_L=t(chol(Q_k))
    l_tilde=solve(Q_L)%*%l_k
    Psi_0=solve(t(Q_L))%*%(l_tilde+rnorm(l+2))

    C_k=t(Psi[[t]][,-k])%*%YB_Grid_Product

    C_tilde=solve(Q_L)%*%t(C_k)
    C_tilde=solve(t(Q_L))%*%C_tilde

    Psi[[t]][, k] = Psi_0 - C_tilde%*%
      solve(C_k %*% C_tilde)%*%C_k%*%Psi_0

    #Norm of f
    temp_norm=as.numeric(t(Psi[[t]][, k])%*%YB_Grid_Product%*%Psi[[t]][, k])
    Psi[[t]][,k]=Psi[[t]][,k]/sqrt(abs(temp_norm))
    Beta[[t]][,k]=Beta[[t]][,k]*sqrt(abs(temp_norm))

    ##Sample Lambda
    lambda[t,k]=rgamma(1,shape=(l+1)/2,rate=t(Psi[[t]][,k])%*%Omega%*%Psi[[t]][,k]/2)
    lambda[t,k]=max(10^(-8),lambda[t,k])
  }

  ####################################
  ##Sample Principal Score: Beta
  ####################################
  for (i in sample(1:n_unit,n_unit)){
    temp_f=YB_matrix[[i]]%*%Psi[[t]]
    f_sq_sum=apply(temp_f^2,2,sum)
    temp_sigma=1/(f_sq_sum/sigma_noise[t-1]+phi[[t-1]][i,]/sigma_K[t-1,])
    for (k in sample(1:K,K))
    {
      temp_mean=
        sum((Obs_value[[i]]-AUG_X[id_index[[i]],]%*%theta[t-1,]-
               YB_matrix[[i]]%*%Psi[[t]][,-k]%*%Beta[[t]][i,-k])*temp_f[,k])/sigma_noise[t-1]+
        (tau_0[t-1,k]*(1-Treatment_Ind[i])+tau_1[t-1,k]*Treatment_Ind[i])*phi[[t-1]][i,k]/sigma_K[t-1,k]
      Beta[[t]][i,k]=rnorm(1,mean=temp_mean*temp_sigma[k],sd=sqrt(temp_sigma[k]))
    }
  }

  ####################################
  #Sample Tau_0,Tau_1
  ####################################
  tau_0_sigma=1/(apply(phi[[t-1]][Treatment_Ind==0,],2,sum)/sigma_K[t-1,]+1/sigma_tau)
  tau_0_mean=apply(Beta[[t]][Treatment_Ind==0,]*phi[[t-1]][Treatment_Ind==0,],2,sum)/sigma_K[t-1,]
  tau_0[t,]=rnorm(n=K,mean=tau_0_mean*tau_0_sigma,sd=sqrt(tau_0_sigma))

  tau_1_sigma=1/(apply(phi[[t-1]][Treatment_Ind==1,],2,sum)/sigma_K[t-1,]+1/sigma_tau)
  tau_1_mean=apply(Beta[[t]][Treatment_Ind==1,]*phi[[t-1]][Treatment_Ind==1,],2,sum)/sigma_K[t-1,]
  tau_1[t,]=rnorm(n=K,mean=tau_1_mean*tau_1_sigma,sd=sqrt(tau_1_sigma))

  ####################################
  #Sample theta
  ####################################
  X_Product=t(AUG_X)%*%AUG_X
  Theta_Sigma=solve(X_Product/sigma_noise[t-1]+diag(rep(1/sigma_theta,ncol(AUG_X))))

  process_value=unlist(lapply(1:n_unit,FUN=function(x){YB_matrix[[x]]%*%Psi[[t]]%*%Beta[[t]][x,]}))
  Theta_Mean=Theta_Sigma%*%(t(AUG_X)%*%(Obs-process_value)/sigma_noise[t-1])

  theta[t,]=mvrnorm(n=1,mu=Theta_Mean,Sigma=Theta_Sigma)

  ####################################
  #Sample Precision/Variance Parameter
  ####################################
  SSE=sum((Obs-process_value-AUG_X%*%theta[t,])^2)
  sigma_noise[t]=1/rgamma(1,shape=n_total/2,rate=SSE/2)

  ####################################
  #Random Effect MGP
  ####################################
  # SSE_Beta=apply(Beta[[t]]-(1-Treatment_Ind)%*%t(tau_0[t,])-Treatment_Ind%*%t(tau_1[t,]),2,
  #                FUN=function(x){sum(x^2)})
  Beta_Error=Beta[[t]]-(1-Treatment_Ind)%*%t(tau_0[t,])-Treatment_Ind%*%t(tau_1[t,])
  #Sample Phi
  phi[[t]]=phi[[1]]
  for (k in sample(1:K,K))
  {
    for (i in sample(1:n_unit,n_unit))
    {
      phi[[t]][i,k]=rgamma(n=1,shape=(v+1)/2,
                           rate=v/2+(Beta_Error[i,k]^2/sigma_K[t-1,k])/2)

    }
  }


  prod=1
  prod_vector=1
  for (k in 2:K)
  {
    prod=prod*rd_delta[t-1,k]
    prod_vector=c(prod_vector,prod)
  }

  rd_delta[t,1]=rgamma(1,shape=ard[t-1,1]+K*n_unit/2,
                       rate=1+0.5*sum(prod_vector*apply(phi[[t]] * Beta_Error^ 2, 2, sum)))

  #K>=2
  shuffle_k=sample(2:K,K-1)
  scan=0
  for (k in shuffle_k)
  {
    prod=1
    for (h in 1:(k-1))
    {
      #Update the sampled one
      if (h%in%c(1,shuffle_k[0:scan]))
      {prod=prod*rd_delta[t,h]
      }else{prod=prod*rd_delta[t-1,h]}

    }
    prod_vector=prod
    if(k<K){
      for (h in (k+1):K)
      {
        if (h%in%shuffle_k[0:scan])
        {prod=prod*rd_delta[t,h]
        }else{prod=prod*rd_delta[t-1,h]}
        prod_vector=c(prod_vector,prod)
      }

      rd_delta[t,k]=rgamma(1,shape=ard[t-1,2]+(K-k+1)*n_unit/2,
                           rate=1+0.5*sum(prod_vector*apply(phi[[t]][,k:K] * Beta_Error[,k:K]^ 2, 2, sum)))
      #print(1+0.5*sum(prod_vector*apply(phi[[t]][,k:K] * Beta_Error[,k:K]^ 2, 2, sum)))
    }
    else{
      rd_delta[t,k]=rgamma(1,shape=ard[t-1,2]+(K-k+1)*n_unit/2,
                           rate=1+0.5*sum(prod_vector*sum(phi[[t]][,K] * Beta_Error[,K]^ 2)))
    }
    scan=scan+1

  }

  #Reconstruct sigma_K
  variance=1
  for (k in 1:K)
  {

    variance=variance/rd_delta[t,k]
    sigma_K[t,k]=variance
  }

  proposal=runif(1,ard[t-1,1]-step_size,ard[t-1,1]+step_size)
  proposal=abs(proposal)
  acceptance_rate=log(rd_delta[t,1])*(proposal-ard[t-1,1])+
    log(gamma(ard[t-1,1]))-log(gamma(proposal))+
    ard[t-1,1]-proposal+log(proposal)-log(ard[t-1,1])
  acceptance_rate=min(exp(acceptance_rate),1)
  if (runif(1)<acceptance_rate)
  {ard[t,1]=proposal}else{ard[t,1]=ard[t-1,1]}


  proposal=runif(1,ard[t-1,2]-step_size,ard[t-1,2]+step_size)
  proposal=max(abs(proposal),2)
  acceptance_rate=sum(log(rd_delta[t,2:K]))*(proposal-ard[t-1,2])+
    (log(gamma(ard[t-1,2]))-log(gamma(proposal)))*(K-1)+
    ard[t-1,2]-proposal+log(proposal)-log(ard[t-1,2])
  acceptance_rate=min(exp(acceptance_rate),1)
  if (runif(1)<acceptance_rate)
  {ard[t,2]=proposal}else{ard[t,2]=ard[t-1,2]}


  if(t%%1000==0){
    print(paste("==",t,"=="))}


}


# Y_MCMC_Result=list(sigma_K=sigma_K,sigma_noise=sigma_noise,
#                    tau_0=tau_0,tau_1=tau_1,Beta=Beta,
#                    Psi=Psi,lambda=lambda,theta=theta,
#                    knots_y=knots_y)
#save(Y_MCMC_Result,file=paste(mediation_index[m],adverse.index[adv_id],"_Y0223.RData",sep=""))

B_Grid_NEW=construct_B(work_grid,knots_y)
pos_sample_size=simu_time/10
burn_in=(simu_time*9/10+1):simu_time
eigen=lapply(1:pos_sample_size,FUN=function(x){B_Grid_NEW%*%(Psi[[burn_in[x]]])})
eigen_mean=apply(simplify2array(eigen),1:2,mean)
eigen_ci_up=apply(simplify2array(eigen),1:2,FUN=function(x){quantile(x,0.975)})
eigen_ci_down=apply(simplify2array(eigen),1:2,FUN=function(x){quantile(x,0.025)})

theta=theta[(simu_time*9/10+1):simu_time,]
beta_coeff=lapply(1:pos_sample_size,FUN=function(x){Beta[[burn_in[x]]]})

# plot(eigen_mean[,1],ylim=range(eigen_mean),type='l',col="red")
# lines(eigen_mean[,2],col="blue")
# lines(eigen_ci_up[,2],type='l',lty=2)
# lines(eigen_ci_down[,2],type='l',lty=2)
# lines(eigen_ci_up[,1],type='l',lty=2)
# lines(eigen_ci_down[,1],type='l',lty=2)
# lines(eigen_mean[,3],col="blue")
# lines(eigen_ci_up[,3],type='l',lty=2)
# lines(eigen_ci_down[,3],type='l',lty=2)

#Uncertainty
#apply(sigma_K[burn_in,],2,mean)

# direct_process=lapply(1:pos_sample_size,FUN=function(x){eigen[[x]]%*%(tau_1[burn_in[x],]-tau_0[burn_in[x],])})
# direct_process_mean=apply(simplify2array(direct_process),1:2,mean)
# direct_process_down=apply(simplify2array(direct_process),1:2,function(x){quantile(x,0.025)})
# direct_process_up=apply(simplify2array(direct_process),1:2,function(x){quantile(x,0.975)})
# 
# pdf("Direct_Effect_Non.pdf",width=8,height=6)
# plot(work_grid,direct_process_mean+0.3,ylim=range(direct_process),
#      type='l',lty=6,lwd=2,main="Direct Effect Estimation\n
#      Scenario 1",xlab="Time",ylab="Effect")
# lines(work_grid,direct_process_up+0.3,lty=2)
# lines(work_grid,direct_process_down+0.3,lty=2)
# # lines(work_grid,y_1_1-y_0_1,lwd=2)
# abline(h=0)
# legend("top",legend=c("True Value","Posterior Mean","95% Credible Interval"),
#        lty=c(1,6,2),lwd=c(2,2,1))
# abline(h=0)##Direct Effect
# dev.off()

##Mediation Effect
# indirect_process=lapply(1:pos_sample_size,FUN=function(x){gamma[x]*mediator_effect[[x]]})
# indirect_process_mean=apply(simplify2array(indirect_process),1:2,mean)
# indirect_process_down=apply(simplify2array(indirect_process),1:2,function(x){quantile(x,0.025)})
# indirect_process_up=apply(simplify2array(indirect_process),1:2,function(x){quantile(x,0.975)})
# total_effect=lapply(1:pos_sample_size,FUN=function(x){indirect_process[[x]]+direct_process[[x]]})
# total_effect_mean=apply(simplify2array(total_effect),1:2,mean)
# total_effect_down=apply(simplify2array(total_effect),1:2,function(x){quantile(x,0.025)})
# total_effect_up=apply(simplify2array(total_effect),1:2,function(x){quantile(x,0.975)})

# pdf("Total_Estimate_Non.pdf",height=4.5,width=8.5)
# par(mar = c(4, 4.5, 2, 1))
# plot(work_grid,direct_process_mean,type='l',ylim=c(0,6.5),xlab="Time",
#      ylab=expression(paste(tau[TE]^t)), main="T=15",lty=6,lwd=2)
# lines(work_grid,direct_process_down,lty=2)
# lines(work_grid,direct_process_,lty=2)
# # lines(work_grid,y_1_1-y_0_0,lwd=2,lty=1)
# legend("top",legend=c("True Value","Posterior Mean","95% Credible Interval"),
#        lty=c(1,6,2),lwd=c(2,2,1))
# dev.off()

