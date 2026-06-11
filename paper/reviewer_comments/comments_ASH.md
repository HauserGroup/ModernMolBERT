# Overleaf Comments Extraction

Extracted 43 messages including replies.
5 messages appear truncated in the saved DOM.

## T001.1 - LaTeX line 159 [possibly truncated]

**Author:** alexshauser

**Time:** 8 June, 9:45 am

**data-pos:** `6125`; **data-top:** ``

**Comment:** Its a well written introduction, but I find the subheaders with single paragraphs a bit unconvention...

**LaTeX offset context:**

```latex
ing code. \end{abstract} % ============================================================ [[POS]]\section{Introduction}% \label{sec:introduction} % =======================================
```

**LaTeX context:**

```latex
\section{Introduction}%
```

## T002.1 - LaTeX line 172

**Author:** alexshauser

**Time:** 8 June, 9:43 am

**data-pos:** `7569`; **data-top:** ``

**Comment:** Maybe Claude can also have the references in order, ie starting from [1

**LaTeX offset context:**

```latex
. The most prominent examples are circular fingerprints such as the Extended Connectivity [[POS]]Fingerprint~(ECFP)~\citep{rogersExtendedConnectivityFingerprints2010}, which hash the pres
```

**LaTeX context:**

```latex
Fingerprint~(ECFP)~\citep{rogersExtendedConnectivityFingerprints2010}, which
```

## T003.1 - LaTeX line 212

**Author:** alexshauser

**Time:** 8 June, 10:39 pm

**data-pos:** `10107`; **data-top:** ``

**Comment:** more informative with parameter and dimension size?

**LaTeX offset context:**

```latex
key design choices alongside \model{}; \cref{sec:related-work} discusses them in detail. [[POS]] % ------ Table 1: molecular embedder comparison --------------- \begin{table}[htbp] \ce
```

## T004.1 - LaTeX line 281

**Author:** alexshauser

**Time:** 8 June, 9:50 am

**data-pos:** `13470`; **data-top:** ``

**Comment:** It's closer to methods and results (implementation) rather than introduction at this point

**LaTeX offset context:**

```latex
odernbert{}-style encoder has not been systematically evaluated as a molecular embedder. [[POS]]\paragraph{Atom Pair Encoding.} Tokenization which is the mapping from a raw string to dis
```

**LaTeX context:**

```latex
\paragraph{Atom Pair Encoding.}
```

## T005.1 - LaTeX line 291

**Author:** alexshauser

**Time:** 8 June, 9:52 am

**data-pos:** `14255`; **data-top:** ``

**Comment:** ref fig 1 before fig 2

**LaTeX offset context:**

```latex
ry is a valid concatenation of complete \selfies{} symbols rather than a character-level a[[POS]]rtefact (\cref{fig:design}C). \ape{} thus occupies a middle ground: compact like subword t
```

**LaTeX context:**

```latex
than a character-level artefact (\cref{fig:design}C). \ape{} thus occupies a
```

## T006.1 - LaTeX line 294

**Author:** alexshauser

**Time:** 8 June, 9:51 am

**data-pos:** `14410`; **data-top:** ``

**Comment:** you don't reference figure 1 anywhere

**LaTeX offset context:**

```latex
d: compact like subword tokenization, chemically principled like atom-level tokenization. [[POS]] \paragraph{Modernising the encoder architecture.} Many molecular string encoders such Che
```

## T007.1 - LaTeX line 303

**Author:** alexshauser

**Time:** 8 June, 9:53 am

**data-pos:** `15113`; **data-top:** ``

**Comment:** maybe needs explanation on why RoPE can solve this limitation

**LaTeX offset context:**

```latex
}-based molecular language modelling. Especially RoPE may be suited to molecular strings, [[POS]]where absolute sequence position is partly an artefact of the chosen graph traversal rathe
```

**LaTeX context:**

```latex
Especially RoPE may be suited to molecular strings, where absolute sequence
```

## T008.1 - LaTeX line 318

**Author:** alexshauser

**Time:** 8 June, 9:45 am

**data-pos:** `15940`; **data-top:** ``

**Comment:** unconventional

**LaTeX offset context:**

```latex
selfies{} embedder is an open question we examine in an ablation (\cref{sec:ablations}). [[POS]]\paragraph{Contributions.} We introduce \model{}, a compact family of encoder-only molecul
```

**LaTeX context:**

```latex
\paragraph{Contributions.}
```

## T009.1 - LaTeX line 330 [possibly truncated]

**Author:** alexshauser

**Time:** 8 June, 9:50 am

**data-pos:** `16754`; **data-top:** ``

**Comment:** THe 'chemically principled like atom-level tokenization.' could be better represented in the APE mer...

**LaTeX offset context:**

```latex
e two molecular design choices that distinguish \model{} from prior \selfies{} encoders. [[POS]]\begin{figure}[htbp] \centering \includegraphics[width=\linewidth]{figures/Fig1_C} \
```

**LaTeX context:**

```latex
\begin{figure}[htbp]
```

## T010.1 - LaTeX line 337

**Author:** alexshauser

**Time:** 8 June, 11:27 pm

**data-pos:** `17078`; **data-top:** ``

**Comment:** After MLM training, can you decode back to chemically valid molecules?

**LaTeX offset context:**

```latex
curated, canonicalised, and converted to \selfies{} strings, then pre-tokenized with [[POS]] the \selfies{}-adapted \ape{} tokenizer (\cref{fig:design}C). A \modernbert{} enco
```

**LaTeX context:**

```latex
the \selfies{}-adapted \ape{} tokenizer (\cref{fig:design}C). A
```

## T011.1 - LaTeX line 342

**Author:** alexshauser

**Time:** 8 June, 11:04 pm

**data-pos:** `17460`; **data-top:** ``

**Comment:** are the hyperparameters explained anywhere?

**LaTeX offset context:**

```latex
dings over non-special \selfies{} tokens are extracted without any gradient updates [[POS]] and passed to lightweight downstream classifiers (ridge, random forest, $k$-nearest
```

**LaTeX context:**

```latex
and passed to lightweight downstream classifiers (ridge, random forest,
```

## T012.1 - LaTeX line 377 [possibly truncated]

**Author:** alexshauser

**Time:** 8 June, 9:54 am

**data-pos:** `19249`; **data-top:** ``

**Comment:** why it's own header? Usually this is integrated into introduction or later discussion. Overlaps with...

**LaTeX offset context:**

```latex
{fig:design} \end{figure} % ============================================================ [[POS]]\section{Related Work}% \label{sec:related-work} % =======================================
```

**LaTeX context:**

```latex
\section{Related Work}%
```

## T013.1 - LaTeX line 382

**Author:** alexshauser

**Time:** 8 June, 9:56 am

**data-pos:** `19363`; **data-top:** ``

**Comment:** panel A looks stretched and both panels low res

**LaTeX offset context:**

```latex
\label{sec:related-work} % ============================================================ [[POS]]\subsection{Molecular String Representations} \label{sec:rel-representations} The chemica
```

**LaTeX context:**

```latex
\subsection{Molecular String Representations}
```

## T014.1 - LaTeX line 385

**Author:** alexshauser

**Time:** 8 June, 9:55 am

**data-pos:** `19442`; **data-top:** ``

**Comment:** reference or your own calculation (methods/results)

**LaTeX offset context:**

```latex
======== \subsection{Molecular String Representations} \label{sec:rel-representations} [[POS]]The chemical-validity guarantee of \selfies{}~\citep{krennSELFIESSelfReferencingEmbedded20
```

**LaTeX context:**

```latex
The chemical-validity guarantee of \selfies{}~\citep{krennSELFIESSelfReferencingEmbedded2020}
```

## T015.1 - LaTeX line 395

**Author:** alexshauser

**Time:** 8 June, 10:00 am

**data-pos:** `20112`; **data-top:** ``

**Comment:** Better suited for discussion imo

**LaTeX offset context:**

```latex
than equivalent \smiles{}-based models~\citep{krennSELFIESSelfReferencingEmbedded2020}. [[POS]]\subsection{Molecular Language Models} \label{sec:rel-lms} The fingerprint ECFP~\citep{ro
```

**LaTeX context:**

```latex
\subsection{Molecular Language Models}
```

## T016.1 - LaTeX line 435 [possibly truncated]

**Author:** alexshauser

**Time:** 8 June, 10:00 am

**data-pos:** `22617`; **data-top:** ``

**Comment:** THis is very true, especially on wether your benchmarked molecules are in the chembl training set (c...

**LaTeX offset context:**

```latex
\citep{liuPretrainingMolecularGraph2022}, and Uni-Mol~\citep{zhouUniMolUniversal3D2023}. [[POS]]These graph-based methods are outside the scope of \model{}, which operates on sequential
```

**LaTeX context:**

```latex
These graph-based methods are outside the scope of \model{}, which
```

## T017.1 - LaTeX line 464

**Author:** alexshauser

**Time:** 8 June, 10:12 pm

**data-pos:** `24339`; **data-top:** ``

**Comment:** maybe best to integrate into intro / discussion

**LaTeX offset context:**

```latex
te-of-the-art results among encoder-only models on natural-language and code benchmarks. [[POS]]\subsection{Molecular Tokenization} \label{sec:rel-tokenization} The atom-preserving \ape
```

**LaTeX context:**

```latex
\subsection{Molecular Tokenization}
```

## T018.1 - LaTeX line 488

**Author:** alexshauser

**Time:** 8 June, 10:13 pm

**data-pos:** `25346`; **data-top:** ``

**Comment:** specify

**LaTeX offset context:**

```latex
024,mayuareAPETokenizer} with several modifications for \selfies{} pretraining at scale. [[POS]]\paragraph{Molecular-aware pre-tokenization.} \selfies{} strings are first split on a rege
```

**LaTeX context:**

```latex
\paragraph{Molecular-aware pre-tokenization.}
```

## T019.1 - LaTeX line 559

**Author:** alexshauser

**Time:** 8 June, 10:17 pm

**data-pos:** `29528`; **data-top:** ``

**Comment:** is it just 10k compounds that get filtered that way?

**LaTeX offset context:**

```latex
ChEMBL~ID, canonical \smiles{}, and \selfies{} in that order) with seed~13, yielding \num{[[POS]]2390317} training and \num{24228} validation molecules. The resulting \selfies{} strings a
```

**LaTeX context:**

```latex
yielding \num{2390317} training and \num{24228} validation molecules.
```

## T020.1 - LaTeX line 607

**Author:** alexshauser

**Time:** 8 June, 10:24 pm

**data-pos:** `32426`; **data-top:** ``

**Comment:** why those specifically?

**LaTeX offset context:**

```latex
l parameters are updated. These embeddings are evaluated with three downstream models --- [[POS]]a ridge classifier, random forests, and $k$-nearest neighbours --- whose hyperparameters a
```

**LaTeX context:**

```latex
embeddings are evaluated with three downstream models --- a ridge classifier,
```

## T021.1 - LaTeX line 634

**Author:** alexshauser

**Time:** 8 June, 10:25 pm

**data-pos:** `33755`; **data-top:** ``

**Comment:** which is? REF?

**LaTeX offset context:**

```latex
Dataset sizes range from 475 to 93\,087 compounds, with task counts ranging from 1 to 27 ([[POS]]SIDER). The full list with per-dataset statistics is given in Table~4 of \citet{praskiBenc
```

**LaTeX context:**

```latex
ranging from 1 to 27 (SIDER).
```

## T022.1 - LaTeX line 640

**Author:** alexshauser

**Time:** 8 June, 10:27 pm

**data-pos:** `33965`; **data-top:** ``

**Comment:** integrate diectly into methods!?

**LaTeX offset context:**

```latex
5}, whose benchmark selection we follow directly. All datasets are evaluated by ROC-AUC. [[POS]]\subsection{Baselines} \label{sec:baselines} We compare \model{} against both fixed and l
```

**LaTeX context:**

```latex
\subsection{Baselines}
```

## T023.1 - LaTeX line 683 [possibly truncated]

**Author:** alexshauser

**Time:** 8 June, 10:33 pm

**data-pos:** `36551`; **data-top:** ``

**Comment:** NIce aggregation figure, but perhaps show an example ROC to understand the underlying aggregation st...

**LaTeX offset context:**

```latex
f molecular embeddings. Throughout, ROC-AUC values are reported $\times100$, so one point [[POS]]corresponds to $0.01$ ROC-AUC. Direct paired comparisons use only jointly evaluated datase
```

**LaTeX context:**

```latex
Throughout, ROC-AUC values are reported $\times100$, so one point corresponds
```

## T024.1 - LaTeX line 686

**Author:** alexshauser

**Time:** 8 June, 10:29 pm

**data-pos:** `36685`; **data-top:** ``

**Comment:** should be continues from 1

**LaTeX offset context:**

```latex
comparisons use only jointly evaluated datasets, reported in \cref{fig:bootstrap-ci} and [[POS]]\cref{tab:bootstrap-cis}. \Cref{tab:main-results} reports mean ROC-AUC by task group; per-
```

**LaTeX context:**

```latex
\cref{tab:bootstrap-cis}.
```

## T025.1 - LaTeX line 690

**Author:** alexshauser

**Time:** 8 June, 10:30 pm

**data-pos:** `36840`; **data-top:** ``

**Comment:** is the underlying data for each dataset and each baseline shown anywhere?

**LaTeX offset context:**

```latex
C-AUC by task group; per-dataset results for all models are given in \cref{tab:pertask}. [[POS]]\begin{figure}[htbp] \centering \includegraphics[width=0.82\linewidth]{figures/bootstr
```

**LaTeX context:**

```latex
\begin{figure}[htbp]
```

## T026.1 - LaTeX line 690

**Author:** alexshauser

**Time:** 8 June, 10:32 pm

**data-pos:** `36843`; **data-top:** ``

**Comment:** would be nice to annotate in figure directly the positive MMB direction

**LaTeX offset context:**

```latex
UC by task group; per-dataset results for all models are given in \cref{tab:pertask}. \be[[POS]]gin{figure}[htbp] \centering \includegraphics[width=0.82\linewidth]{figures/bootstrap_
```

**LaTeX context:**

```latex
\begin{figure}[htbp]
```

## T027.1 - LaTeX line 710

**Author:** alexshauser

**Time:** 8 June, 10:42 pm

**data-pos:** `37785`; **data-top:** ``

**Comment:** why is there a difference in number of datasets again?

**LaTeX offset context:**

```latex
} \input{tables/main_results_table} \model{}-base attains a mean ROC-AUC of 77.4 across [[POS]]24 evaluated benchmark datasets, and \model{}-small attains 77.3 across all 25 datasets (p
```

**LaTeX context:**

```latex
\model{}-base attains a mean ROC-AUC of 77.4 across 24 evaluated benchmark
```

## T028.1 - LaTeX line 726

**Author:** alexshauser

**Time:** 8 June, 10:48 pm

**data-pos:** `38850`; **data-top:** ``

**Comment:** could that be stated in the initial table for comparison?

**LaTeX offset context:**

```latex
its clearest advantage (77.6 vs.\ \model{}-base 70.7), consistent with its much larger and[[POS]] more chemically diverse pretraining corpus. This pattern is consistent with frozen learne
```

**LaTeX context:**

```latex
larger and more chemically diverse pretraining corpus. This pattern is
```

## T029.1 - LaTeX line 726

**Author:** alexshauser

**Time:** 8 June, 10:48 pm

**data-pos:** `38896`; **data-top:** ``

**Comment:** REF

**LaTeX offset context:**

```latex
e 70.7), consistent with its much larger and more chemically diverse pretraining corpus. T[[POS]]his pattern is consistent with frozen learned embeddings transferring best to tasks govern
```

**LaTeX context:**

```latex
larger and more chemically diverse pretraining corpus. This pattern is
```

## T030.1 - LaTeX line 758

**Author:** alexshauser

**Time:** 8 June, 10:59 pm

**data-pos:** `41008`; **data-top:** ``

**Comment:** not all of these are global properties though

**LaTeX offset context:**

```latex
olarity, flexibility, aromaticity, and size, indicating that the embedding space captures [[POS]]global physicochemical structure even though it is never given these properties as trainin
```

**LaTeX context:**

```latex
indicating that the embedding space captures global physicochemical structure
```

## T031.1 - LaTeX line 760

**Author:** alexshauser

**Time:** 8 June, 10:51 pm

**data-pos:** `41108`; **data-top:** ``

**Comment:** veeery small font, should be closer to figure legend size

**LaTeX offset context:**

```latex
sicochemical structure even though it is never given these properties as training signal. [[POS]] \begin{figure}[htbp] \centering \includegraphics[width=\linewidth]{figures/Fig_baseli
```

## T032.1 - LaTeX line 761

**Author:** alexshauser

**Time:** 8 June, 10:53 pm

**data-pos:** `41109`; **data-top:** ``

**Comment:** its the underlying for Figure 3 if I am not mistaken, should it come first then?

**LaTeX offset context:**

```latex
icochemical structure even though it is never given these properties as training signal. [[POS]]\begin{figure}[htbp] \centering \includegraphics[width=\linewidth]{figures/Fig_baselin
```

**LaTeX context:**

```latex
\begin{figure}[htbp]
```

## T033.1 - LaTeX line 796

**Author:** alexshauser

**Time:** 8 June, 10:56 pm

**data-pos:** `42703`; **data-top:** ``

**Comment:** y axis labels are very small

**LaTeX offset context:**

```latex
arity and reported in \cref{tab:pertask}. }% \label{fig:group-bars} \end{figure} [[POS]]\begin{figure}[htbp] \centering \includegraphics[width=\linewidth]{figures/Fig_4} \c
```

**LaTeX context:**

```latex
\begin{figure}[htbp]
```

## T034.1 - LaTeX line 805

**Author:** alexshauser

**Time:** 8 June, 10:55 pm

**data-pos:** `43158`; **data-top:** ``

**Comment:** spell out as QED (Quantitative Estimate of Drug-likeness)

**LaTeX offset context:**

```latex
nt represents one molecule; colour indicates the corresponding property value: ALogP, [[POS]]QED, polar surface area, rotatable bonds, aromatic rings, and heavy atom count. Pa
```

**LaTeX context:**

```latex
property value: ALogP, QED, polar surface area, rotatable bonds, aromatic
```

## T035.1 - LaTeX line 810

**Author:** alexshauser

**Time:** 8 June, 10:57 pm

**data-pos:** `43461`; **data-top:** ``

**Comment:** so whats the interpretation

**LaTeX offset context:**

```latex
are not chemically meaningful, and only relative neighbourhoods and broad spatial [[POS]] trends should be interpreted. }% \label{fig:embedding-space} \end{figure} \subsectio
```

**LaTeX context:**

```latex
trends should be interpreted.
```

## T036.1 - LaTeX line 851

**Author:** alexshauser

**Time:** 8 June, 11:01 pm

**data-pos:** `45670`; **data-top:** ``

**Comment:** more of a supplementary figure

**LaTeX offset context:**

```latex
gure}[htbp] \centering \includegraphics[width=\linewidth]{figures/Fig_2} \caption{% [[POS]] \textbf{Internal comparison of the four \model{} checkpoints on downstream ROC-AUC
```

**LaTeX context:**

```latex
\textbf{Internal comparison of the four \model{} checkpoints on downstream
```

## T037.1 - LaTeX line 910

**Author:** alexshauser

**Time:** 8 June, 11:05 pm

**data-pos:** `48544`; **data-top:** ``

**Comment:** funny wording

**LaTeX offset context:**

```latex
ar representation learning rather than as a claim about all molecular foundation models. [[POS]]It is important to read every number in this paper as a \emph{frozen-embedding} result. No
```

**LaTeX context:**

```latex
It is important to read every number in this paper as a \emph{frozen-embedding}
```

## T038.1 - LaTeX line 930

**Author:** alexshauser

**Time:** 8 June, 11:06 pm

**data-pos:** `49782`; **data-top:** ``

**Comment:** REF.

**LaTeX offset context:**

```latex
\model{} is competitive with, but does not surpass, the strong ECFP4 fingerprint baseline [[POS]]and trails MoLFormer, which is pre-trained on roughly $400\times$ more molecules. The cont
```

**LaTeX context:**

```latex
and trails MoLFormer, which is pre-trained on roughly $400\times$ more
```

## T039.1 - LaTeX line 930

**Author:** alexshauser

**Time:** 8 June, 11:07 pm

**data-pos:** `49815`; **data-top:** ``

**Comment:** could you train MMB on ZINC instead?

**LaTeX offset context:**

```latex
does not surpass, the strong ECFP4 fingerprint baseline and trails MoLFormer, which is pr[[POS]]e-trained on roughly $400\times$ more molecules. The contribution is therefore a compact,
```

**LaTeX context:**

```latex
and trails MoLFormer, which is pre-trained on roughly $400\times$ more
```

## T040.1 - LaTeX line 935

**Author:** alexshauser

**Time:** 8 June, 11:07 pm

**data-pos:** `50070`; **data-top:** ``

**Comment:** wording

**LaTeX offset context:**

```latex
hile substantially improving on prior string-based language models at comparable scale. [[POS]]A practical advantage of our is accessibility. ModernMolBERT is released in the standard H
```

**LaTeX context:**

```latex
A practical advantage of our is accessibility. ModernMolBERT is released in the standard Hugging Face Transformers format and can be used with the current Python ecosystem without model-specific runtime dependencies, legacy Python versions, custom inference kernels, or external chemistry toolkits at embedding time once inputs are provided as SELFIES strings. Frozen embeddings can therefore be extracted with a few lines of standard Transformers code, making the model easier to incorporate into existing screening, retrieval, and benchmark pipelines than models requiring specialised environments. Likewise, the standalone tokenizer is straightforward to use in a similar fashion.
```

## T041.1 - LaTeX line 943

**Author:** alexshauser

**Time:** 8 June, 11:13 pm

**data-pos:** `51403`; **data-top:** ``

**Comment:** could one train on the same corpus to compare the models more directly?

**LaTeX offset context:**

```latex
specialised inference infrastructure. \subsection{Limitations} \label{sec:limitations} [[POS]]\model{} is pre-trained exclusively on \chembl{}~36, a corpus dominated by drug-like, Lipi
```

**LaTeX context:**

```latex
\model{} is pre-trained exclusively on \chembl{}~36, a corpus
```

## T042.1 - LaTeX line 944

**Author:** alexshauser

**Time:** 8 June, 11:18 pm

**data-pos:** `51465`; **data-top:** ``

**Comment:** what about peptides :D

**LaTeX offset context:**

```latex
} \label{sec:limitations} \model{} is pre-trained exclusively on \chembl{}~36, a corpus [[POS]]dominated by drug-like, Lipinski-compliant compounds. Compared with MoLFormer, which draws
```

**LaTeX context:**

```latex
dominated by drug-like, Lipinski-compliant compounds.
```

## T043.1 - LaTeX line 983

**Author:** alexshauser

**Time:** 8 June, 11:19 pm

**data-pos:** `53871`; **data-top:** ``

**Comment:** YEs. How does it scale?

**LaTeX offset context:**

```latex
ing-encoder scope of this work. The most direct is broader pretraining beyond \chembl{}, i[[POS]]ncluding PubChem, ZINC, Enamine, or other sources that better cover natural products, frag
```

**LaTeX context:**

```latex
pretraining beyond \chembl{}, including PubChem, ZINC, Enamine, or other
```
