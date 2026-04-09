# 15418-final-project

<!--
\newcommand{\hwname}{15-418 Final Project Proposal}
\documentclass{article}
\usepackage[left=1in,right=1in,top=1in,bottom=1in]{geometry}
\usepackage{graphicx} % Required for inserting images
% \usepackage{fancyhdr}
\usepackage{hyperref}
\usepackage{amsmath, amssymb}
\usepackage{caption}
\usepackage{listings, xcolor, lstautogobble}
\usepackage{enumerate}
\usepackage{algorithm, algorithmic}
\usepackage{biblatex}
\addbibresource{cite.bib}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
    pdftitle={Overleaf Example},
    pdfpagemode=FullScreen,
}
\definecolor{background_color}{RGB}{255, 255, 255}
\definecolor{string_color}    {RGB}{180, 156,   0}
\definecolor{identifier_color}{RGB}{  0,   0,   0}
\definecolor{keyword_color}   {RGB}{ 64, 100, 255}
\definecolor{comment_color}   {RGB}{  0, 117, 110}
\definecolor{number_color}    {RGB}{ 84,  84,  84}
\lstdefinestyle{code}{
    basicstyle=\ttfamily,
    numberstyle=\tiny\ttfamily\color{number_color},
    backgroundcolor=\color{background_color},
    stringstyle=\color{string_color},
    identifierstyle=\color{identifier_color},
    keywordstyle=\color{keyword_color},
    commentstyle=\color{comment_color},
    numbers=left,
    frame=none,
    columns=fixed,
    tabsize=2,
    breaklines=true,
    keepspaces=true,
    showstringspaces=false,
    captionpos=b,
    autogobble=true,
    mathescape=true,
    literate={~}{{$\thicksim$}}1
             {~=}{{$\eeq$}}1
}
\newcommand{\code}[1]{\lstinline[style=code]`#1`}
\title{Parallel Remeshing}
\author{Ryan Kpamegan (rkpamega), Junbo Huang (junboh)}
\date{\href{https://github.com/rkpamegan/15418-final-project}{Project Page}}

\begin{document}
\maketitle
\section{Summary}
We propose a parallel implementation of a remeshing algorithm using CUDA. The program will take modify a mesh to be more uniform.

\section{Background}
In computer graphics, a mesh is the surface of a 3D object, defined by its vertices and edges, along with faces, which are finite planes enclosed by a set of vertices and edges. In general, "good" meshes are regular: they have vertices with similar neighborhood size, edges with similar lengths, and faces with roughly equal angles. There are of course exceptions, but these properties are the target most of the time. 
\begin{figure}[h!]
    \centering
    \includegraphics[width=0.4\linewidth]{background_avg.png}
    \caption{Vertex position averaging \\ Source: 15-362 Fall 2025}
    \label{fig:background-avg}
\end{figure}
\begin{figure}[h!]
    \centering
    \includegraphics[width=0.4\linewidth]{background_split.png}
    \includegraphics[width=0.4\linewidth]{background_collapse.png}
    \caption{Splitting long edges and collapsing short edges \\ Source: 15-362 Fall 2025}
    \label{fig:background-edge}
\end{figure} \\
A remeshing algorithm takes in a mesh as input and manipulates its elements to give it the desired properties. The algorithm we will use can add or remove edges, and move vertices, as shown in the above figures. This is very computationally heavy, as each vertex and edge in the entire mesh must be iterated over. As the level of detail on a mesh increases, so does the number of objects, and thus the remeshing becomes more expensive. When the algorithm is isotropic the same steps are applied to every vertex and edge indiscriminately, which lends itself to parallelism and can make remeshing much faster.


\section{The Challenge}
The biggest challenge to implementing this project is the race conditions for modifying the graph underlying the mesh. For example, when adjusting edges, we have the option to split the edge into two and add a vertex in between, thus halving the edge length. This requires updating the surrounding vertices and faces so that they include the new edges. Similarly, we can collapse the edge, pulling its two incident vertices into one, which also requires updates to surrounding elements. If done naively in parallel, this will cause data races, where some elements can refer to others that don't exist, or fail to refer to newly created elements. Locality introduces an even bigger problem, as we would like to keep nearby elements in the cache for later computation while also keeping them far apart enough to not affect each other. 

Divergent execution also presents a problem for the algorithm. As mentioned above, edges can be either split or removed during remeshing. Thus, it is possible to lose lots of performance when many edges undergo different operations. We define length thresholds during the routine for which operation should be performed on an edge (or whether or not it is performed at all); the value of these thresholds can impact scaling due to this divergent execution.

\section{Resources}
We plan to mainly use Fall 2025 15-362 Computer Graphics lecture notes \cite{graphics} and assignments as a base for the structure and logic of the algorithm. We may use test cases from the assignments for our own implementation. We will also use Assignment 2 from this course as a reference for writing CUDA code.

\section{Goals and Deliverables}
The goal we plan to achieve is the implement parallel isotropic remeshing, with a speedup of 6-7x on 8 cores. The goal we hope to achieve if all goes well is another remeshing algorithm that uses Centroidal Voronoi Diagrams \cite{valette}, although we have not yet evaluated its potential for parallelism.

\section{Platform Choice}
The project will be implemented using CUDA on the GHC machines. Isotropic remeshing (or at least most of it) is inherently SIMD and thus CUDA is a natural choice.  The GHC machines are easily accessible and relatively inexpensive to use, allowing us to rapidly develop our program.

\section{Schedule}
\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|}
        \hline
        Week of & Item \\
        \hline
        4/6 & Establish base code, generate small test cases  \\
        \hline
        4/13 & Begin parallel implementation  \\
        \hline
        4/20 & Finish parallel implementation  \\
        \hline
        4/27 & Optimize performance \\
        \hline
    \end{tabular}
\end{table}
\newpage
\printbibliography
\end{document}
-->