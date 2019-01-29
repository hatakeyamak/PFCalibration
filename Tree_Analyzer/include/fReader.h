#ifndef fReader_h
#define fReader_h

#include <iostream>
#include <vector>

#include "TROOT.h"
#include "TF1.h"
#include "TMath.h"
#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TMath.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TCanvas.h"
#include "TDirectory.h"
#include "TBranch.h"
#include "TString.h"
#include "TStyle.h"
#include "TInterpreter.h"
#include "TStyle.h"
#include "TLorentzVector.h"

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>

namespace globalTChain
{
 
  TTreeReader  fReader;  //!the tree reader
 
  //
  // Set up TTreeReader's
  // -- use MakeSelector of root
  //
  // Readers to access the data (delete the ones you do not need).
  
  TTreeReaderArray<float> true_   = {fReader, "true"};
  TTreeReaderArray<float> p_      = {fReader, "p"};
  TTreeReaderArray<float> ecal_   = {fReader, "ecal"};
  TTreeReaderArray<float> hcal_   = {fReader, "hcal"};
  TTreeReaderArray<float> ho_     = {fReader, "ho"};
  TTreeReaderArray<float> eta_    = {fReader, "eta"};
  TTreeReaderArray<float> phi_    = {fReader, "phi"};
  TTreeReaderArray<int>   charge_ = {fReader, "charge"};

  TTreeReaderArray<float> *dr_    = {fReader, "dr"};
  TTreeReaderArray<float> *Eecal_ = {fReader, "Eecal"};
  TTreeReaderArray<float> *Ehcal_ = {fReader, "Ehcal"};
  TTreeReaderArray<float> *pfcID_ = {fReader, "pfcID"};
  //TTreeReaderArray<int> pfcs_ = {fReader, "pfcs"};

  //TTreeReaderValue<long> run_       = {fReader, "run"};
  //TTreeReaderValue<long> evt_       = {fReader, "evt"};
  //TTreeReaderValue<long> lumiBlock_ = {fReader, "lumiBlock"};
  //TTreeReaderValue<long> time_      = {fReader, "time"};

}

#endif
