/*
 * Copyright © 2017 Gert Wollny
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "st_tests_common.h"

#include <tgsi/tgsi_ureg.h>
#include <tgsi/tgsi_info.h>
#include <mesa/program/prog_instruction.h>

#include <gtest/gtest.h>
#include <utility>
#include <algorithm>
#include <iostream>

using std::vector;


/* Test two arrays life time simple */
TEST_F(LifetimeEvaluatorExactTest, TwoArraysSimple)
{
   const vector<FakeCodeline> code = {
      { TGSI_OPCODE_MOV , {MT(1, 1, WRITEMASK_XYZW)}, {MT(0, in0, "")}, {}, ARR()},
      { TGSI_OPCODE_MOV , {MT(2, 1, WRITEMASK_XYZW)}, {MT(0, in1, "")}, {}, ARR()},
      { TGSI_OPCODE_ADD , {MT(0,out0, WRITEMASK_XYZW)}, {MT(1,1,"xyzw"), MT(2,1,"xyzw")}, {}, ARR()},
      { TGSI_OPCODE_END}
   };
   run (code, array_lt_expect({{0,2, WRITEMASK_XYZW}, {1,2, WRITEMASK_XYZW}}));
}

/* Test two arrays life time simple */
TEST_F(LifetimeEvaluatorExactTest, TwoArraysSimpleSwizzleX_Y)
{
   const vector<FakeCodeline> code = {
      { TGSI_OPCODE_MOV , {MT(1, 1, WRITEMASK_X)}, {MT(0, in0, "")}, {}, ARR()},
      { TGSI_OPCODE_MOV , {MT(2, 1, WRITEMASK_Y)}, {MT(0, in1, "")}, {}, ARR()},
      { TGSI_OPCODE_ADD , {MT(0,out0,1)}, {MT(1,1,"x"), MT(2,1,"y")}, {}, ARR()},
      { TGSI_OPCODE_END}
   };
   run (code, array_lt_expect({{0, 2, WRITEMASK_X}, {1, 2, WRITEMASK_Y}}));
}

/* Test array written before loop and read inside, must survive the loop */
TEST_F(LifetimeEvaluatorExactTest, ArraysWriteBeforLoopReadInside)
{
   const vector<FakeCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in1}, {}},
      { TGSI_OPCODE_MOV, {MT(1, 1, WRITEMASK_X)}, {MT(0, in0, "")}, {}, ARR()},
      { TGSI_OPCODE_BGNLOOP },
      { TGSI_OPCODE_ADD, {MT(0,1, WRITEMASK_X)}, {MT(1,1,"x"), {MT(0,1, "x")}}, {}, ARR()},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, array_lt_expect({{1, 4, WRITEMASK_X}}));
}

/* Test array written conditionally in loop must survive the whole loop */
TEST_F(LifetimeEvaluatorExactTest, ArraysConditionalWriteInNestedLoop)
{
   const vector<FakeCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in1}, {}},
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_BGNLOOP },
      {     TGSI_OPCODE_IF, {}, {1}, {}},
      {       TGSI_OPCODE_MOV, {MT(1, 1, WRITEMASK_Z)}, {MT(0, in0, "")}, {}, ARR()},
      {     TGSI_OPCODE_ENDIF },
      {     TGSI_OPCODE_ADD, {MT(0,1, WRITEMASK_X)}, {MT(1,1,"z"), {MT(0,1, "x")}}, {}, ARR()},
      {   TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, array_lt_expect({{1, 8, WRITEMASK_Z}}));
}

/* Test array written conditionally in loop must survive the whole loop */
TEST_F(LifetimeEvaluatorExactTest, ArraysConditionalWriteInNestedLoop2)
{
   const vector<FakeCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in1}, {}},
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_BGNLOOP },
      {     TGSI_OPCODE_IF, {}, {1}, {}},
      {       TGSI_OPCODE_BGNLOOP },
      {         TGSI_OPCODE_MOV, {MT(1, 1, WRITEMASK_Z)}, {MT(0, in0, "")}, {}, ARR()},
      {       TGSI_OPCODE_ENDLOOP },
      {     TGSI_OPCODE_ENDIF },
      {     TGSI_OPCODE_ADD, {MT(0,1, WRITEMASK_X)}, {MT(1,1,"z"), {MT(0,1, "x")}}, {}, ARR()},
      {   TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, array_lt_expect({{1, 10, WRITEMASK_Z}}));
}


/* Test distinct loops */
TEST_F(LifetimeEvaluatorExactTest, ArraysReadWriteInSeparateScopes)
{
   const vector<FakeCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in1}, {}},
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_MOV, {MT(1, 1, WRITEMASK_W)}, {MT(0, in0, "")}, {}, ARR()},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_ADD, {MT(0,1, WRITEMASK_X)}, {MT(1,1,"w"), {MT(0,1, "x")}}, {}, ARR()},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, array_lt_expect({{2, 6, WRITEMASK_W}}));
}
