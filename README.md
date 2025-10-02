# Neurovariety
Symbolic neuro-variety toolkit (Magma)


NeuroVarieties.m – symbolic neuro-variety toolkit (Magma)

Overview

A compact library to build symbolic feed-forward networks, extract the

coefficient map, impose gauges, and study the dimension of the resulting

“neuro-varieties” via Jacobian-rank sampling. Includes a small search

harness and a composite-Veronese + projection constructor.

Main entry points

• BuildNetworkDataNamed(n,d)

• ParametrizationNamed(n,d)

• AffineGaugeParametrizationNamed(n,d)

• GaugeJacobianRankRandom(n,d)

• NeuroVarietyStats(n,d : tries := 10)

• ExhaustiveSmallDefectTest  (top-level blocks at the end of the file)

• Composite-Veronese and Projection Construction (final section)

Conventions

• n = [n0,…,nL]      layer widths (inputs n0, outputs nL)

• d = [d1,…,d_{L-1}] coordinate-wise power activations x ↦ x^{di}

• K                  base field (defaults to Q via RationalField())

• Gauge (affine)     last column of each W_i is set to 1

Quick start

SetColumns(1000);

n := [2,2,3,1];  d := [3,3];

phi, _ := ParametrizationNamed(n,d);

phiAff, _ := AffineGaugeParametrizationNamed(n,d);

rank, P := GaugeJacobianRankRandom(n,d);

stats := NeuroVarietyStats(n,d : tries := 20);

Notes

• This file is self-contained; no external packages required.

• All maps return readable names for domain/codomain coordinates.

• Jacobian evaluation samples random integer points and avoids poles.
