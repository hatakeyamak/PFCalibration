// ------------------------------------------------------------------------------------
//  ROOT macro that produces average RecHit energy from PFG ntuples
//
//  Author : Ken H
//  Written on May 24, 2018
// ------------------------------------------------------------------------------------
//  
// Pre-requisite :
//
//   You should have the PFG ntuple for the Run from which you want to do a measurement. 
//   Instruction on how to make PFG ntuples can be found here : FIXME link here 
//
//   You should have "Fig" directory for plots 
//
// Usage : 
//
//   $ root -b  
//   root> .L ana_PFStudy.C+
//   root> ana_PFStudy("/cms/data/store/user/hatake/HCAL/ntuples/10_2_x/pi50_trees_MCfull_CMSSW_10_2_0_pre3_*.root","hcal_timestudy_pi50_histograms.root")
//   or
//   root> ana_PFStudy("list_trees_pi50_MCfull_CMSSW_10_2_0_pre3.txt","hcal_timestudy_pi50_histograms.root")
//   or
//   from command line:
/*
     root.exe -b -q 'ana_PFStudy.C++("trees_relval_ttbar_phase2_age_new2_4500ultimate.root","hcal_noisestudy_histograms_age_new2_4500ultimate.root")'
     root.exe -b -q 'ana_PFStudy.C++("trees_relval_ttbar_phase2_age_org.root","hcal_noisestudy_histograms_age_org.root")'
     root.exe -b -q 'ana_PFStudy.C++("trees_relval_ttbar_phase2_noage.root","hcal_noisestudy_histograms_noage.root")'
 */
//    
// -----------------------------------------------------------------------------------
// 

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip> // for setw()
#include <algorithm> 

//#include "../include/fReader.h" // Read from TTRee

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

// In order to use vector of vectors : vector<vector<data type> >
// ACLiC makes dictionary for this
// [ref] http://root.cern.ch/phpBB3/viewtopic.php?f=3&t=10236&p=44117#p44117
#ifdef __MAKECINT__
#pragma link C++ class std::vector < std::vector<int> >+;
#pragma link C++ class std::vector < std::vector<float> >+;
#endif

using namespace std;
//using namespace globalTChain;

bool DRAWPLOTS  = false;  // draw plots or not (make "Fig" directory first before turning this on)
bool VERBOSE    = false;  // print out mean +/- sigma for each channel or not

// Assemble a list of inputfiles

std::vector<std::string> GetInputFiles(TString sampleType)
{
  int numFiles = 4;
  std::string path = "/eos/uscms/store/user/bcaraway/SinglePi/";
  std::string startName = "_trees_";
  std::string endName = ".root";
  std::vector<std::string> inputFiles;
  
  for( int iFile = 1 ; iFile<=numFiles ; iFile++ )
    {
      std::ostringstream fileName;
      fileName << path<< sampleType << startName << iFile << endName;
      inputFiles.push_back(fileName.str());
    }
  return inputFiles;
}
//
// Book histograms
//

void book1D(TList *v_hist, std::string name, int n, double min, double max);
void book1DProf(TList *v_hist, std::string name, int n, double min, double max, double ymin, double ymax, Option_t *option);
void book2D(TList *v_hist, std::string name, int xn, double xmin, double xmax, int yn, double ymin, double ymax); // added by Bryan
void bookHistograms(TList *v_hist);
//
// Fill histograms
//
void fill1D(TList *v_hist, std::string name, double value);
void fill1D(TList *v_hist, std::string name, double value, double valuey);
void fill1DProf(TList *v_hist, std::string name, double value, double valuey);
void fill2D(TList *v_hist, std::string name, double valuex, double valuey);  // added by Bryan 
//
// Aux
//
void relabelProfA(TList *v_hist, std::string name);
void relabel2D   (TList *v_hist, std::string name); // added by Bryan
//
// Main analyzer
//
void PFCheckRun(std::vector<std::string> inputFiles, TString outfile, int maxevents, int option=2) 
{ 
   cout << "[PF analyzer] Running option " << option << " for " << endl; 
   // fit pannel display option
   gStyle->SetOptFit(1011);
   //
   // Get the tree from the PFG ntuple 
   //
   TChain *ch = new TChain("s");

   for (unsigned int iFile=0; iFile<inputFiles.size(); ++iFile) {
    ch->Add(inputFiles[iFile].c_str());
    std::cout<<inputFiles[iFile]<<std::endl;
   }
   printf("%d;\n",ch->GetNtrees());
   printf("%lld;\n",ch->GetEntries());

   //fReader.SetTree(ch);  //the tree reader (Defined in fReader.h)
   TTree* sTree;
   sTree = (TTree*)ch;
   Float_t true_, p_, eta_, ecal_, hcal_, ho_;
   TBranch *b_true, *b_p, *b_eta, *b_ecal, *b_hcal, *b_ho;
   //std::vector<float> *Eecal_;
   //std::vector<float> *Ehcal_;
   //std::vector<float> *dr_;
   
   //TBranch        *b_E_ecal,*b_E_hcal,*b_dr;
   //Eecal_ = 0;
   //Ehcal_ = 0;
   //dr_ = 0;
   sTree->SetMakeClass(1);
   if(sTree->GetBranchStatus("true"))
     sTree->SetBranchAddress("true", &true_, &b_true);
   sTree->SetBranchAddress("p", &p_, &b_p);
   sTree->SetBranchAddress("eta", &eta_, &b_eta);
   sTree->SetBranchAddress("ecal", &ecal_, &b_ecal);
   sTree->SetBranchAddress("hcal", &hcal_, &b_hcal);
   sTree->SetBranchAddress("ho", &ho_, &b_ho);
   //sTree->SetBranchAddress("Eecal", &Eecal_, &b_E_ecal);
   //sTree->SetBranchAddress("Ehcal", &Ehcal_, &b_E_hcal);
   //sTree->SetBranchAddress("dr", &dr_, &b_dr);


   
   //
   // Define histograms to fill
   //
   TList *v_hist = new TList();
   bookHistograms(v_hist); // most of histograms booked here
   //
   // Loop over entries
   //
   unsigned int nentries = (Int_t)ch->GetEntries();
   cout << "[HGCal analyzer] The number of entries is: " << nentries << endl;
   //
   // Set up output tree
   //
   TTree t1("t1","a simple Tree with simple variables");
   Float_t gen_e, p, eta,pf_ecalRaw, pf_hcalRaw, pf_hoRaw, pf_totalRaw;
   t1.Branch("gen_e",  &gen_e,  "gen_e/F");
   t1.Branch("p",  &p,  "p/F");
   t1.Branch("eta",  &eta,  "eta/F");
   t1.Branch("pf_ecalRaw",&pf_ecalRaw,"pf_ecalRaw/F");
   t1.Branch("pf_hcalRaw",&pf_hcalRaw,"pf_hcalRaw/F");
   t1.Branch("pf_hoRaw",&pf_hoRaw,"pf_hoRaw/F");
   t1.Branch("pf_totalRaw",&pf_totalRaw,"pf_totalRaw/F");
   
   //---------------------------------------------------------------------------------------------------------
   // main event loop
   //---------------------------------------------------------------------------------------------------------
   								  
   
   //int ievent=0;
   std::string strtmp;
   //while (fReader.Next()) {
   for (int ievent =0 ; ievent < sTree->GetEntries(); ievent++){
     // Progress indicator 
     //ievent++;
     sTree->GetEntry(ievent);
     if(ievent%100000==0) cout << "[HCAL analyzer] Processed " << ievent << " out of " << nentries << " events" << endl; 
     if (maxevents>0 && ievent>maxevents) Break;
     if (true_>400 || true_<=5.0) continue; // to handle bug with gen_e > 400 GeV
     //--------------------
     // Loop over Gen
     //--------------------
     //std::cout<<"true: "<<true_<<"\t ecal: "<<ecal_<<"\t hcal: "<<hcal_<<std::endl;
     gen_e = true_;
     p = p_;
     eta = fabs(eta_);
     pf_ecalRaw = ecal_;
     pf_hcalRaw = hcal_;
     pf_hoRaw = ho_;
     pf_totalRaw = ecal_+hcal_+ho_;
     
     if (eta < 1.5){
       strtmp = "Response_barrel";
       fill1DProf(v_hist, strtmp, gen_e, (pf_totalRaw - gen_e)/gen_e);
     }
     if (eta > 1.5 && eta < 3.0){
       strtmp = "Response_endcap";
       fill1DProf(v_hist, strtmp, gen_e, (pf_totalRaw - gen_e)/gen_e);
     }
     strtmp = "Response_eta";
     fill1DProf(v_hist, strtmp, eta, (pf_totalRaw - gen_e)/gen_e); 
     
     t1.Fill();
   }
   // Event loop ends
   //---------------------------------------------------------------------------------------------------------
   // main event loop ends
   //---------------------------------------------------------------------------------------------------------

   // output file for histograms
   std::cout<<"Output file: \t"<<outfile<<std::endl;
   TFile file_out(outfile,"RECREATE");

   t1.Write();
   v_hist->Write();
   
   file_out.ls();
   file_out.Close();

}

//
// Main function
//
//void ana_PFStudy(TString rootfile="../../HGCalTreeMaker/test/ttbar_10_4_D30_pt25.root",TString outfile="pfstudy_histograms.root",int maxevents=-1)
// "D30" for D30 geo, "D28" for D28
void ana_main(TString sampleType, TString testFile = "test_numEvent10000.root")
{
  int maxevents=-1;
  // edit 
  bool test_file = false; // if testing setup with single file (will have to edit below for file choice)
  TString outfile  = sampleType+"_histos_trees_test.root";

  std::vector<std::string> inputFiles;
  if (!test_file){
      inputFiles = GetInputFiles(sampleType);
    }
  else inputFiles.push_back(static_cast<std::string>(testFile));

  PFCheckRun(inputFiles, outfile, maxevents, 0);
}

void ana_PFStudy(std::vector<TString> sampleType = {"singlePi"}){ // loop over HGCal geometry configurations
  for (int i = 0; i != sampleType.size(); i++){
    ana_main(sampleType[i]);
  }
}
 

//
// --- Aux ---
//

//
// Book 1D histograms
//
void book1D(TList *v_hist, std::string name, int n, double min, double max)
{
  TH1D *htemp = new TH1D(name.c_str(), name.c_str(), n, min, max);
  v_hist->Add(htemp);
}
//
// Book 1D profile histograms
//
void book1DProf(TList *v_hist, std::string name, int n, double min, double max, double ymin, double ymax, Option_t *option="")
{
  TProfile *htemp = new TProfile(name.c_str(), name.c_str(), n, min, max, ymin, ymax, option);
  v_hist->Add(htemp);
}
//
// Book 2D profile histograms 
//
void book2D(TList *v_hist, std::string name, int xn, double xmin, double xmax, int yn, double ymin, double ymax) 
{
  TH2D *htemp = new TH2D(name.c_str(), name.c_str(), xn, xmin, xmax, yn, ymin, ymax);
  v_hist->Add(htemp);
}
//
// Book histograms
//
void bookHistograms(TList *v_hist)
{

  Char_t histo[100];
  std::string strtmp;
  
  //
  // Booking histograms
  // 

  book1DProf(v_hist, "Response_barrel", 50, 0, 500, -0.5, 0.5);
  book1DProf(v_hist, "Response_endcap", 50, 0, 500, -0.5, 0.5);
  book1DProf(v_hist, "Response_eta", 30, 0, 3, -0.2, 0.2);
  
}
//
// relabel 1DProf histograms
//
void relabelProfA(TList *v_hist, std::string name)
{
  TProfile* htemp = (TProfile*) v_hist->FindObject(name.c_str());
  htemp->GetXaxis()->SetBinLabel(1,"Track");
  htemp->GetXaxis()->SetBinLabel(2,"ECAL");
  htemp->GetXaxis()->SetBinLabel(3,"HCAL");
  htemp->GetXaxis()->SetBinLabel(5,"HCAL d1");
  htemp->GetXaxis()->SetBinLabel(6,"HCAL d2");
  htemp->GetXaxis()->SetBinLabel(7,"HCAL d3");
  htemp->GetXaxis()->SetBinLabel(8,"HCAL d4");
  htemp->GetXaxis()->SetBinLabel(9,"HCAL d5");
  htemp->GetXaxis()->SetBinLabel(10,"HCAL d6");
  htemp->GetXaxis()->SetBinLabel(11,"HCAL d7");
}

void relabel2D(TList *v_hist, std::string name) // added by Bryan 
{
  TH2D* htemp = (TH2D*) v_hist->FindObject(name.c_str());
  htemp->GetXaxis()->SetBinLabel(1,"Charged Had");
  htemp->GetXaxis()->SetBinLabel(2,"Electron");
  htemp->GetXaxis()->SetBinLabel(4,"Photon");
  htemp->GetXaxis()->SetBinLabel(5,"Nuetral Had");
}
//
// Fill 1D histograms
//
void fill1D(TList *v_hist, std::string name, double value)
{
  TH1F* htemp = (TH1F*) v_hist->FindObject(name.c_str());
  htemp->Fill(value);
}
void fill1D(TList *v_hist, std::string name, double value, double valuey)
{
  TH1F* htemp = (TH1F*) v_hist->FindObject(name.c_str());
  htemp->Fill(value, valuey);
}
//
// Fill 1D Profile histograms
//
void fill1DProf(TList *v_hist, std::string name, double value, double valuey)
{
  TProfile* htemp = (TProfile*) v_hist->FindObject(name.c_str());
  htemp->Fill(value,valuey);
}
// 
// Fill 2D histograms
//
void fill2D(TList *v_hist, std::string name, double valuex, double valuey)
{
  TH2D* h_temp = (TH2D*) v_hist->FindObject(name.c_str());
  h_temp->Fill(valuex,valuey);
}
