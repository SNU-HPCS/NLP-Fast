#!/bin/bash
BASEDIR=$(dirname $0)
BUILDDIR=${BASEDIR}/../model/build/
VERI_SMALLSET=${BASEDIR}/../verification/smallset
VERI_WEIGHT=${BASEDIR}/../verification/weight

TARGETS=(
	"baseline.none"
	#"baseline.clflush"
	#"baseline.prefetch"
	"partial_head.none"
	#"partial_head.clflush"
	#"partial_head.prefetch"
	"column.none"
	#"column.clflush"
	#"column.prefetch"
	"all_opt.none"
	#"all_opt.clflush"
	#"all_opt.prefetch"
)

for target in "${TARGETS[@]}"; do
	echo "Start to verify ${target}_t1_b1"
	${BUILDDIR}/${target} -m 1 ${VERI_SMALLSET} ${VERI_WEIGHT} -c 64 > /tmp/${target}_t1_b1
	cat /tmp/${target}_t1_b1 | grep "difference"

	echo "Start to verify ${target}_t1_b2"
	${BUILDDIR}/${target} -t 1 -b 2 -m 1 ${VERI_SMALLSET} ${VERI_WEIGHT} -c 64 > /tmp/${target}_t1_b2
	cat /tmp/${target}_t1_b2 | grep "difference"

	echo "Start to verify ${target}_t1_b4"
	${BUILDDIR}/${target} -t 1 -b 4 -m 1 ${VERI_SMALLSET} ${VERI_WEIGHT} -c 64 > /tmp/${target}_t1_b4
	cat /tmp/${target}_t1_b4 | grep "difference"

	echo "Start to verify ${target}_t1_b8"
	${BUILDDIR}/${target} -t 1 -b 8 -m 1 ${VERI_SMALLSET} ${VERI_WEIGHT} -c 64 > /tmp/${target}_t1_b8
	cat /tmp/${target}_t1_b8 | grep "difference"

	echo "Start to verify ${target}_t16_b1"
	${BUILDDIR}/${target} -t 16 -b 1 -m 1 ${VERI_SMALLSET} ${VERI_WEIGHT} -c 64 > /tmp/${target}_t16_b1
	cat /tmp/${target}_t16_b1 | grep "difference"

	echo "Start to verify ${target}_t16_b2"
	${BUILDDIR}/${target} -t 16 -b 2 -m 1 ${VERI_SMALLSET} ${VERI_WEIGHT} -c 64 > /tmp/${target}_t16_b2
	cat /tmp/${target}_t16_b2 | grep "difference"

	echo "Start to verify ${target}_t8_b4"
	${BUILDDIR}/${target} -t 8 -b 4 -m 1 ${VERI_SMALLSET} ${VERI_WEIGHT} -c 64 > /tmp/${target}_t8_b4
	cat /tmp/${target}_t8_b4 | grep "difference"

	echo "Start to verify ${target}_t8_b1"
	${BUILDDIR}/${target} -t 8 -b 1 -m 1 ${VERI_SMALLSET} ${VERI_WEIGHT} -c 64 > /tmp/${target}_t8_b1
	cat /tmp/${target}_t8_b1 | grep "difference"

	echo "Start to verify ${target}_t4_b4"
	${BUILDDIR}/${target} -t 4 -b 4 -m 1 ${VERI_SMALLSET} ${VERI_WEIGHT} -c 64 > /tmp/${target}_t4_b4
	cat /tmp/${target}_t4_b4 | grep "difference"

	echo "Start to verify ${target}_t4_b1"
	${BUILDDIR}/${target} -t 4 -b 1 -m 1 ${VERI_SMALLSET} ${VERI_WEIGHT} -c 64 > /tmp/${target}_t4_b1
	cat /tmp/${target}_t4_b1 | grep "difference"

	echo "Start to verify ${target}_t2_b4"
	${BUILDDIR}/${target} -t 2 -b 4 -m 1 ${VERI_SMALLSET} ${VERI_WEIGHT} -c 64 > /tmp/${target}_t2_b4
	cat /tmp/${target}_t2_b4 | grep "difference"

	echo "Start to verify ${target}_t2_b1"
	${BUILDDIR}/${target} -t 2 -b 1 -m 1 ${VERI_SMALLSET} ${VERI_WEIGHT} -c 64 > /tmp/${target}_t2_b1
	cat /tmp/${target}_t2_b1 | grep "difference"
done
