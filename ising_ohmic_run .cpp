//三回目のテスト

// ising_ohmic_test2 ... ohmic, remove boost
// ising_ohmic_test3 ... use of mpich, gather data
// ising_ohmic_test4 ... fourier test
// ising_ohmic_test5 ... fourier transformation + pade
// ising_ohmic_run ... for running jobs to obtain corr. in time
// use of openmpi
// compile: openmpic++ ****.cpp
// run:     openmpirun -np (# of proc.) ****.cpp
#include "MersenneTwister.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <mpi.h>
using namespace std;
//using namespace boost;

class cmdline {
private:
  int nint,ndouble;
  vector<int*> int_pointers;
  vector<double*> double_pointers;
  //
  string help_header;
  vector<string> type;
  vector<int> number;
  vector<string> keys,comments;
  //
  vector<string> command_line;
public:
  cmdline(const string header) : help_header(header) {
    nint = 0;
    ndouble = 0;
  }
  void add(const string key, const string comment, int &param) {
    keys.push_back(key);
    comments.push_back(comment);
    type.push_back("int");
    int_pointers.push_back(&param);
    number.push_back(nint);
    nint++;
  }
  void add(const string key, const string comment, double &param) {
    keys.push_back(key);
    comments.push_back(comment);
    type.push_back("double");
    double_pointers.push_back(&param);
    number.push_back(ndouble);
    ndouble++;
  }
  void help() {
    cout << help_header << endl;
    for (int i=0; i<keys.size(); i++) {
      cout << keys[i] << " (" << type[i] <<"): " << comments[i] << endl;
    }
  }
  void parse(const int argc, char** argv) {
    if (argc <= 1) {
      help();
      exit(1);
    }
    for (int i=1; i<=argc-1; i++) {
      command_line.push_back(argv[i]);
    }
    for (int i=0; i<keys.size(); i++) {
      for (int j=0; j<command_line.size(); j++) {
	if (keys[i] == command_line[j]) {
	  if (type[i] == "int") {
	    *int_pointers[number[i]] = atoi(command_line[j+1].c_str());
	  } else if (type[i] == "double") {
	    *double_pointers[number[i]] = atof(command_line[j+1].c_str());
	  } 
	  break;
	} 
	if (j == command_line.size() - 1) {
	  help();
	  exit(1);
	}
      }
    }
  }
};

template<class T> inline double sqr(T x) { 
  double xtmp = (double) x;
  return xtmp*xtmp; }
template<class T> inline double quo(T x) { 
  double xtmp = (double) x;
  return xtmp*xtmp*xtmp*xtmp; }

struct parameters {
  int argc;
  char** argv;
  // parameters given by a command line
  int nsite;
  int nthermal;
  int nmeasure,nbin;
  int seed;
  double Delta, delta_s;
  void output(int k) {
    cout << " nsite " << nsite
	 << " Delta " << Delta
	 << " delta_s " << delta_s
	 << " seed " << seed
	 << " nthermal " << nthermal
	 << " nbin " << nbin
	 << " measure " << k+1 << " of " << nmeasure
	 << " :";
  }
};

class observable {
  double sum_tmp, sum_av, sum_sq;
  int ndata;
public:
  observable() {
    ndata = 0;
    sum_av = 0.0;
    sum_sq = 0.0;
  }
  ~observable() {}
  void operator<<(double x) {
    ndata++;
    sum_av += x;
    sum_sq += sqr(x);
  }
  double mean() {
    return ndata > 0 ? sum_av/(double) ndata : 0.0;
  }
  double err() {
    return ndata > 1 ? 
      sqrt((sum_sq/(double)ndata - sqr(mean()))/(double)(ndata - 1)) : 0.0;
  }
};

class Ising_MC_method {
private:
  vector<int> spin,cluster;
  vector<double> J,cumulative;
  // for correlation function
  vector<int> flipped_spin;
  vector<double> distribution;
  // paremters
  parameters p;
  // random generator
  MTRand rng;
public:
  Ising_MC_method(const parameters q) :
    rng((unsigned long) q.seed), p(q) {   

    // exchange interaction
    J.reserve(p.nsite);
    // probability distribution
    cumulative.reserve(p.nsite);
    // spin configration    
    spin.reserve(p.nsite);
    for (int i=0; i<p.nsite; i++) {
      spin[i] = 1;
    }
    // for calculation of correlation functions
    distribution.reserve(p.nsite);
    // for cluster update
    cluster.reserve(p.nsite);
    flipped_spin.reserve(p.nsite);
    // set exchange interaction (s=1 or 3)
    double J_LR, J_NN;
    // ohmic case
    J_LR = p.delta_s;
    J_NN = - log(0.5*p.Delta) - (1. + 0.577215665)*J_LR;
    J[0] = 0.0;
    for (int i=1; i<p.nsite; i++) {
      double x = M_PI*(double)i/(double)p.nsite;
      J[i] = 0.5*J_LR*V_LR(x);
    }
    J[1] += 0.5*J_NN;
    J[p.nsite-1] += 0.5*J_NN;
    // calculate cumulative distribution
    double sum_J = 0.0;
    cumulative[0] = 0.0;
    for (int i=1; i<p.nsite; i++) {
      sum_J += 2.0*J[i];
      // probability of i-th spin-flip with no spin-flip for j<i
      cumulative[i] = 1 - exp(-sum_J); 
    }
  }
  inline double V_LR(double x) {
    return sqr(M_PI/(double)p.nsite)/sqr(sin(x));
    /*
    return quo(M_PI/(double)p.nsite)*
      (2.0 + 4.0*sqr(cos(x)))/6./quo(sin(x));
    */
  }
  void update() {
    // update by Luijten algorithm    
    cluster.resize(0);
    flipped_spin.resize(0);

    int current_spin_initial = int(rng()*p.nsite);
    int sigma = spin[current_spin_initial];
    spin[current_spin_initial] = - spin[current_spin_initial];
    flipped_spin.push_back(current_spin_initial);
    cluster.push_back(current_spin_initial);

    while (cluster.size() > 0) {
      int current_spin = cluster.back();
      cluster.pop_back();
      
      // activate bonds
      int j=0;
      while (true) {
	// bisection search (start)
	double g = rng()*(1.0-cumulative[j]) + cumulative[j];
	if (cumulative[p.nsite-1] < g) break;
	int k_s = j+1;
	int k_e = p.nsite;
	while (k_e - k_s > 1) {
	  int k_tmp = (k_s + k_e)/2;
	  if (cumulative[k_tmp-1] < g) {
	    k_s = k_tmp;
	  } else {
	    k_e = k_tmp;
	  }
	}
	int k = k_s;
	// bisection search (end)
	j = k;
 	int i = (current_spin + k) % p.nsite;
	if (spin[i] == sigma) {
	  spin[i] = - spin[i];
	  flipped_spin.push_back(i);
	  cluster.push_back(i);
	}      
      }
    }
    // spin correlation
    for (int i=0; i<p.nsite; i++) distribution[i] = 0.0;
    for (int i=0; i<flipped_spin.size(); i++) {
      for (int j=0; j<flipped_spin.size(); j++) {
	distribution[(p.nsite + flipped_spin[i] - flipped_spin[j]) % p.nsite] += 1.;
      }
    }
    for (int i=0; i<p.nsite; i++) distribution[i] /= (double)flipped_spin.size();
  }
  double get_corr_func(const int k) {    
    return (double)distribution[k];
  }
  int get_spin(const int i) {
    return spin[i];
  }
};

struct mpi1darray {
  int nsize,nproc,id;
  double *array;
  vector<vector<double> > data;
  mpi1darray(int np, int id_in, vector<double> d) {
    nproc = np;
    id = id_in;
    nsize = d.size();    
    data.resize(np);
    for (int k=0; k<np; k++) {
      data[k].resize(nsize);
    }
    array = new double[nsize];
    for (int j=0; j<nsize; j++) {
      array[j] = d[j];
    } 
  }
  ~mpi1darray() {
    delete[] array;
  }
  void send_receive() {
    if (id == 0) {
      for (int i=0; i<nsize; i++) {
	data[0][i] = array[i];
      }
      for (int k=1; k<nproc; k++) {
	int tag = 1001;
	MPI_Status status;
	MPI_Recv(array,nsize,MPI_DOUBLE,k,tag,
		 MPI_COMM_WORLD,&status);
	for (int j=0; j<nsize; j++) {
	  data[k][j] = array[j];
	}
      }
    } else {
      int tag = 1001;
      MPI_Send(array,nsize,MPI_DOUBLE,0,tag,MPI_COMM_WORLD);
    }
  }
  void get_av(vector<double> &output) {
    output.resize(nsize);
    for (int j=0; j<nsize; j++) {
      double sum = 0.;
      for (int k=0; k<nproc; k++) {
	sum += data[k][j]; 
      } 
      output[j] = sum/(double)nproc;
    }
  }
  // for check
  /*
  void get_err(vector<double> &output) {
    output.resize(nsize);
    vector<double> tmp(nsize);
    for (int j=0; j<nsize; j++) {
      double sum = 0.;
      for (int k=0; k<nproc; k++) {
	sum += data[k][j]; 
      } 
      tmp[j] = sum/(double)nproc;
    }
    for (int j=0; j<nsize; j++) {
      double sum = 0.;
      for (int k=0; k<nproc; k++) {
	sum += sqr(data[k][j]-tmp[j]); 
      } 
      output[j] = sqrt(sum)/(double)nproc;
    }
  }
  */
  void get_sqav(vector<double> &output) {
    output.resize(nsize);
    for (int j=0; j<nsize; j++) {
      double sum = 0.;
      for (int k=0; k<nproc; k++) {
	sum += data[k][j]*data[k][j]; 
      } 
      output[j] = sqrt(sum/(double)nproc);
    }
  }
};

int main(int argc, char** argv) {
  // for parallel computing
  MPI_Init(&argc, &argv);
  int id,nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  // physical parameters
  parameters p;
  // please choose one of two method for parameters
  bool commandline_input = true;
  if (commandline_input) {
    cmdline cmd("Options");  
    cmd.add("nthermal","# of MCS for thermalization",p.nthermal);
    cmd.add("nbin","# of bins",p.nbin);
    cmd.add("nmeasure","# of measurement",p.nmeasure);
    // MCS = nbin*nmeasure
    cmd.add("delta","tunneling matrix",p.Delta);
    cmd.add("nsite","# of sites (=beta)",p.nsite);
    cmd.add("alpha","dissipation strength",p.delta_s);
    cmd.add("nseed","seed of random number generator",p.seed);
    cmd.parse(argc,argv);
  } else {
    p.nthermal = 1000;
    p.nbin = 5000;
    p.nmeasure = 20;
    p.Delta = 0.2;
    p.nsite = 256;
    p.delta_s = 0.2;
    p.seed = 1221;
  }
  p.seed += id*2;

  for (p.delta_s = 0.2; p.delta_s<0.6; p.delta_s+=5.) {
    // cluster update simulation
    Ising_MC_method mc(p); 
    for (int i=0; i<p.nthermal; i++) {
      mc.update();
    }    
    vector<observable> corr2(p.nsite,observable());
    observable sus;
    for (int k=0; k<p.nmeasure; k++) {
      //    observable energy;
      vector<observable> corr(p.nsite,observable());
      for (int kk=0; kk<p.nbin; kk++) {
	mc.update();
	for (int j=0; j<p.nsite; j++) {
	  corr[j] << mc.get_corr_func(j);
	}
      }
      double sum = 0.;
      for (int j=0; j<p.nsite; j++) {
	corr2[j] << corr[j].mean();
	sum += corr[j].mean();
      }
      sus << sum;
    }
    
    vector<double> result_mean(p.nsite+1),result_err(p.nsite+1);
    // vector<double> result_err_check(p.nsite+1);
    for (int j=0; j<p.nsite; j++) {
      result_mean[j] = corr2[j].mean();
      result_err[j] = corr2[j].err();
    }
    result_mean[p.nsite] = sus.mean();
    result_err[p.nsite] = sus.err();
    
    // collect data
    mpi1darray result_mean_mpi(nproc,id,result_mean);
    mpi1darray result_err_mpi(nproc,id,result_err);
    result_mean_mpi.send_receive();
    result_err_mpi.send_receive();
    // output
    if (id == 0) {
      result_mean_mpi.get_av(result_mean);
      //result_mean_mpi.get_err(result_err_check);
      result_err_mpi.get_sqav(result_err);
      for (int j=0; j<p.nsite; j++) {
	cout << j << ' ' 
	  //    << result_err_check[j] << ' ' 
	     << result_mean[j] << ' ' 
	     << result_err[j]/((double)nproc) << endl;
      }
    }
  }
  MPI_Finalize();
}

