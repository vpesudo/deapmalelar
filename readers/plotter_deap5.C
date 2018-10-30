#include "RAT/DS/Root.hh"
#include "RAT/DS/CAL.hh"
#include "RAT/DS/PMT.hh"
#include "RAT/DEAPStyle.hh"
#include "TCanvas.h"
#include "TFile.h"
#include "TTree.h"
#include "TH2F.h"
#include "TLegend.h"
#include <iostream>
#include <fstream>

void ShowFiles(){
 system("ls test*/*");
}
void DeapPlot5(string ifilename) {

  int showmbr = 1;
//  TH2F* h2_r_pred = new TH2F("h_r_pred",";radius (mm);isNeck?",200,0,2000,3,-0.25,1.25);
  TGraph* g_r_pred = new TGraph();

  TGraph* g_r_test = new TGraph();
//  g_r_test->GetXaxis()->SetTitle("radius (mm)");
//  g_r_test->GetYaxis()->SetTitle("isNeckEvent?");
  TGraph* g_r_err  = new TGraph();
  TGraph* g_ch_err  = new TGraph();
  TH2F* hmat = new TH2F("error matrix","error matrix",3,-0.25,1.25,15,-0.25,1.25);
  TH1F* hsigacc = new TH1F("sigacc","sig acceptance",100,0,1);
  TH1F* hbgacc  = new TH1F("bgacc","bg acceptance",100,0,1);
  TGraph* g_sig_bg  = new TGraph();
  TGraph* g_mbr_err = new TGraph();


  TH2F* hr_ch;

  if (showmbr ==0) hr_ch = new TH2F("rad_charge","rad_charge",100,0,1500,120,40,260);
  else hr_ch = new TH2F("rad_charge","rad_charge",100,0,1500,100,0,1500);

  float c1; //R
  float c2; //testY
  float c3; //predY
  float c4; //Charge
  float c5; //MBLikelihood 

 // ifstream fin(ifilename.c_str());
  TString ifile = ifilename.c_str();
  ifstream fin(ifile);
  float badness = 0;
  float badness0 = 0;
  float badness1 = 0;
  int i = 0;
  int neck = 0;
  while (!fin.eof()){
    fin >> c1 >> c2 >> c3 >> c4 >> c5;
    g_r_pred->SetPoint(i,c1,c3); // r versus pred
    g_r_test->SetPoint(i,c1,c2); // r versus test
    g_r_err->SetPoint(i,c1,c2-c3); // r versus error
    g_ch_err->SetPoint(i,c4,c2-c3); // r versus error
    hmat->Fill(c2,c3);
    if (showmbr ==0) hr_ch->Fill(c1,c4);
    else hr_ch->Fill(c1,c5);

    g_mbr_err->SetPoint(i,c5,c2-c3);

    if (c2 == 0) hsigacc->Fill(c3); 
    if (c2 == 1) hbgacc->Fill(c3); 

    badness += abs(c2-c3);
    if(c2==1){
       badness1 += abs(c2-c3);
       neck ++;
    }else      badness0 += abs(c2-c3);
    i++;
  }
  
  cout << Form("bad %f out of %d\n", badness,i);
  badness /= float(i);
  cout << badness*100 << "%\n";

  cout << Form("bad %f out of %d neck\n", badness1,neck);
  badness1 /= float(neck);
  cout << badness1*100 << "%\n";

  // Now we'll draw our plot using the official DEAP-3600 style.
  RAT::DEAPStyle *fStyle = RAT::DEAPStyle::GetStyle();
  fStyle->chooseDSPalette(1); // Palette '1' is a purple colour
  TCanvas* canvas = new TCanvas("cPmtCharge","PMT charge",1400,1100);
  canvas->Divide(3,3);

  float offt = 2.5;

  canvas->cd(1);
  hmat->Draw("colz");
  hmat->GetXaxis()->SetTitle("True");
  hmat->GetYaxis()->SetTitle("Pred");
  hmat->GetXaxis()->SetTitleOffset(offt);
  hmat->GetYaxis()->SetTitleOffset(offt);

  canvas->cd(2);
  g_r_pred->Draw();
  g_r_pred->GetYaxis()->SetRangeUser(-0.2,1.2);
  g_r_pred->GetXaxis()->SetTitle("radius (mm)");
  g_r_pred->GetYaxis()->SetTitle("isNeckEvent?");
  g_r_pred->GetXaxis()->SetTitleOffset(offt);
  g_r_pred->GetYaxis()->SetTitleOffset(offt);
  g_r_pred->SetMarkerStyle(6);
  g_r_pred->Draw("P");
  g_r_test->SetMarkerStyle(6);
  g_r_test->SetMarkerColor(kRed);
  g_r_test->Draw("PSame");
  //fStyle->drawDEAP(2); // '2' means to add the word "Simulation"

   TLegend* legend = new TLegend(0.12,0.9,0.4,0.75);
   legend->AddEntry(g_r_pred,"Prediction","P");
   legend->AddEntry(g_r_test,"True","P");
   legend->Draw();

  canvas->cd(3);
  g_r_err->Draw();
  g_r_err->GetXaxis()->SetTitle("radius (mm)");
  g_r_err->GetYaxis()->SetTitle("err (isNeck-pred)");
  g_r_err->GetYaxis()->SetTitleOffset(offt);
  g_r_err->GetXaxis()->SetTitleOffset(offt);
  g_r_err->SetMarkerStyle(6);
  g_r_err->Draw("P");


  hsigacc->Scale(1./float(i-neck));
  hbgacc->Scale(1./float(neck));

  TH1 * hsigaccC = hsigacc->GetCumulative();
  TH1 * hbgaccC = hbgacc->GetCumulative();
  canvas->cd(4);
  int nbins = hsigaccC->GetNbinsX();
  float bestval = 10.;
  int   bestin  = 200;
  float ratio=0.;

  for (int j=0;j<nbins;j++)
  {
    g_sig_bg->SetPoint(j,hsigaccC->GetBinContent(j),hbgaccC->GetBinContent(j));
    ratio = hbgaccC->GetBinContent(j)/hsigaccC->GetBinContent(j);
    if (hsigaccC->GetBinContent(j)>0.5 && ratio < bestval){
	 bestval = ratio;
	 bestin  = j;
    }
  }
  cout << Form("best ratio from %.4f is %.4f, from sigAcc = %.3f and bg acc = %.3f\n",(float)bestin/100.,bestval, hsigaccC->GetBinContent(bestin),hbgaccC->GetBinContent(bestin)); 
  g_sig_bg->Draw();
  g_sig_bg->GetXaxis()->SetTitle("signal acc");
  g_sig_bg->GetYaxis()->SetTitle("bg acc");
  g_sig_bg->GetYaxis()->SetTitleOffset(offt);
  g_sig_bg->GetXaxis()->SetTitleOffset(offt);
  g_sig_bg->SetMarkerStyle(6);
  g_sig_bg->Draw("P");

  canvas->cd(5);
  hsigaccC->GetXaxis()->SetTitle("Pred");
  hsigaccC->GetYaxis()->SetTitle("signal acc");
  hsigaccC->GetYaxis()->SetTitleOffset(offt);
  hsigaccC->GetXaxis()->SetTitleOffset(offt);
  hsigaccC->Draw();

  canvas->cd(6);
  hbgaccC->GetXaxis()->SetTitle("Pred");
  hbgaccC->GetYaxis()->SetTitle("Bg acc");
  hbgaccC->GetYaxis()->SetTitleOffset(offt);
  hbgaccC->GetXaxis()->SetTitleOffset(offt);
  hbgaccC->Draw();

  canvas->cd(7);
  g_ch_err->Draw();
  g_ch_err->GetXaxis()->SetTitle("charge (QPE)");
  g_ch_err->GetYaxis()->SetTitle("err (isNeck-pred)");
  g_ch_err->GetYaxis()->SetTitleOffset(offt);
  g_ch_err->GetXaxis()->SetTitleOffset(offt);
  g_ch_err->SetMarkerStyle(6);
  g_ch_err->Draw("P");

  canvas->cd(8);
  hr_ch->Draw("colz");
  hr_ch->GetXaxis()->SetTitle("R");
  if(showmbr ==0) hr_ch->GetYaxis()->SetTitle("QPE");
  else hr_ch->GetYaxis()->SetTitle("mbR");
  hr_ch->GetXaxis()->SetTitleOffset(offt);
  hr_ch->GetYaxis()->SetTitleOffset(offt);

  canvas->cd(9);
  g_mbr_err->Draw();
  g_mbr_err->GetXaxis()->SetTitle("MB_R-True R/R");
  g_mbr_err->GetYaxis()->SetTitle("err (isNeck-pred)");
  g_mbr_err->GetYaxis()->SetTitleOffset(offt);
  g_mbr_err->GetXaxis()->SetTitleOffset(offt);
  //g_mbr_err->GetXaxis()->SetRangeUser(-2.,2.);
  g_mbr_err->SetMarkerStyle(6);
  g_mbr_err->Draw("P");


  canvas->Update();
}
