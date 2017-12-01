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

using namespace tgsi_array_merge;

/* Test two arrays life time simple */
TEST_F(LifetimeEvaluatorExactTest, TwoArraysSimple)
{
   const vector<FakeCodeline> code = {
      { TGSI_OPCODE_MOV , {MT(1, 1, WRITEMASK_XYZW)}, {MT(0, in0, "")}, {}, ARR()},
      { TGSI_OPCODE_MOV , {MT(2, 1, WRITEMASK_XYZW)}, {MT(0, in1, "")}, {}, ARR()},
      { TGSI_OPCODE_ADD , {MT(0,out0, WRITEMASK_XYZW)}, {MT(1,1,"xyzw"), MT(2,1,"xyzw")}, {}, ARR()},
      { TGSI_OPCODE_END}
   };
   run (code, array_lt_expect({{1,2,0,2, WRITEMASK_XYZW}, {2,2,1,2, WRITEMASK_XYZW}}));
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
   run (code, array_lt_expect({{1, 2, 0, 2, WRITEMASK_X}, {2, 2, 1, 2, WRITEMASK_Y}}));
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
   run (code, array_lt_expect({{1, 1, 1, 4, WRITEMASK_X}}));
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
   run (code, array_lt_expect({{1, 1, 1, 8, WRITEMASK_Z}}));
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
   run (code, array_lt_expect({{1, 1, 1, 10, WRITEMASK_Z}}));
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
   run (code, array_lt_expect({{1, 1, 2, 6, WRITEMASK_W}}));
}

using SwizzleRemapTest=testing::Test;

TEST_F(SwizzleRemapTest, ArrayRemappingBase_x_x)
{
   array_remapping map1(10, 1, 1);
   ASSERT_EQ(map1.get_target_array_id(), 10u);
   ASSERT_EQ(map1.map_writemask(1), 2);
   ASSERT_EQ(map1.map_one_swizzle(0), 1);
   ASSERT_EQ(map1.combined_swizzle(), 3);
}

TEST_F(SwizzleRemapTest, ArrayRemappingBase_xy_x)
{
   array_remapping map1(5, 3, 1);
   ASSERT_EQ(map1.get_target_array_id(), 5u);
   ASSERT_EQ(map1.map_writemask(1), 4);
   ASSERT_EQ(map1.map_one_swizzle(0), 2);
   ASSERT_EQ(map1.combined_swizzle(), 0x7);
}

TEST_F(SwizzleRemapTest, ArrayRemappingBase_no_reswizzle)
{
   array_remapping map1(5);
   ASSERT_EQ(map1.get_target_array_id(), 5u);
   for (int i = 1; i < 16; ++i)
      ASSERT_EQ(map1.map_writemask(i), i);

   for (int i = 0; i < 4; ++i)
      ASSERT_EQ(map1.map_one_swizzle(i), i);
}


TEST_F(SwizzleRemapTest, ArrayRemappingBase_xyz_x)
{
   array_remapping map1(5, 7, 1);
   ASSERT_EQ(map1.get_target_array_id(), 5u);
   ASSERT_EQ(map1.map_writemask(1), 8);
   ASSERT_EQ(map1.map_one_swizzle(0), 3);
   ASSERT_EQ(map1.combined_swizzle(), 0xF);
}

TEST_F(SwizzleRemapTest, ArrayRemappingBase_xy_xy)
{
   array_remapping map1(5, 3, 3);
   ASSERT_EQ(map1.get_target_array_id(), 5u);
   ASSERT_EQ(map1.map_writemask(1), 4);
   ASSERT_EQ(map1.map_writemask(2), 8);
   ASSERT_EQ(map1.map_writemask(3), 0xC);
   ASSERT_EQ(map1.map_one_swizzle(0), 2);
   ASSERT_EQ(map1.map_one_swizzle(1), 3);
   ASSERT_EQ(map1.combined_swizzle(), 0xF);
}

TEST_F(SwizzleRemapTest, ArrayRemappingBase_xz_xw)
{
   array_remapping map1(5, 5, 9);
   ASSERT_EQ(map1.get_target_array_id(), 5u);
   ASSERT_EQ(map1.map_writemask(1), 2);
   ASSERT_EQ(map1.map_writemask(8), 8);
   ASSERT_EQ(map1.map_writemask(9), 0xA);
   ASSERT_EQ(map1.map_one_swizzle(0), 1);
   ASSERT_EQ(map1.map_one_swizzle(3), 3);
   ASSERT_EQ(map1.combined_swizzle(), 0xF);
}

using ArrayMergeTest=testing::Test;

TEST_F(ArrayMergeTest, ArrayMergeTwoSwizzles)
{
   vector<array_lifetime> alt = {
      {1, 4, 1, 5, WRITEMASK_X},
      {2, 4, 2, 5, WRITEMASK_X},
   };

   vector<array_remapping> expect = {
      {},
      {1, WRITEMASK_X, WRITEMASK_X},
   };

   vector<array_remapping> result(alt.size() + 1);

   get_array_remapping(2, &alt[0], &result[0]);

   EXPECT_EQ(result[1], expect[0]);
   EXPECT_EQ(result[2], expect[1]);

}

TEST_F(ArrayMergeTest, ArrayMergeFourSwizzles)
{
   vector<array_lifetime> alt = {
      {1, 8, 1, 7, WRITEMASK_X},
      {2, 7, 2, 7, WRITEMASK_X},
      {3, 6, 3, 7, WRITEMASK_X},
      {4, 5, 4, 7, WRITEMASK_X},
   };

   vector<array_remapping> expect = {
      {},
      {1, WRITEMASK_X, WRITEMASK_X},
      {1, WRITEMASK_XY, WRITEMASK_X},
      {1, WRITEMASK_XYZ, WRITEMASK_X},
   };

   vector<array_remapping> result(alt.size() + 1);

   get_array_remapping(4, &alt[0], &result[0]);

   EXPECT_EQ(result[1], expect[0]);
   EXPECT_EQ(result[2], expect[1]);
   EXPECT_EQ(result[3], expect[2]);
   EXPECT_EQ(result[4], expect[3]);

}


TEST_F(ArrayMergeTest, SimpleChainMerge)
{
   vector<array_lifetime> input = {
      {1, 3, 1, 5, WRITEMASK_XYZW},
      {2, 2, 6, 7, WRITEMASK_XYZW},
   };

   vector<array_remapping> expect = {
      {},
      {1},
   };

   vector<array_remapping> result(3);
   get_array_remapping(2, &input[0], &result[0]);

   for (unsigned i = 0; i < expect.size(); ++i)
      EXPECT_EQ(result[i + 1], expect[i]);
}

TEST_F(ArrayMergeTest, MergeAndInterleave)
{
   vector<array_lifetime> input = {
      {1, 5, 1, 5, WRITEMASK_X},
      {2, 4, 6, 7, WRITEMASK_X},
      {3, 3, 1, 5, WRITEMASK_X},
      {4, 2, 6, 7, WRITEMASK_X},
   };

   vector<array_remapping> expect = {
      {},
      {1},
      {1, WRITEMASK_X, WRITEMASK_X},
      {1, WRITEMASK_X, WRITEMASK_X}
   };
   vector<array_remapping> result(input.size() + 1);
   get_array_remapping(input.size(), &input[0], &result[0]);

   for (unsigned i = 0; i < expect.size(); ++i)
      EXPECT_EQ(result[i + 1], expect[i]);
}

TEST_F(ArrayMergeTest, MergeAndInterleave2)
{
   vector<array_lifetime> input = {
      {1, 5, 1, 5, WRITEMASK_X},
      {2, 4, 6, 7, WRITEMASK_X},
      {3, 3, 1, 8, WRITEMASK_XY},
      {4, 2, 6, 7, WRITEMASK_X},
   };

   vector<array_remapping> expect = {
      {},
      {1},
      {1, WRITEMASK_X, WRITEMASK_XY},
      {1, WRITEMASK_XYZ, WRITEMASK_X}
   };
   vector<array_remapping> result(input.size() + 1);
   get_array_remapping(input.size(), &input[0], &result[0]);

   for (unsigned i = 0; i < expect.size(); ++i)
      EXPECT_EQ(result[i + 1], expect[i]);
}


TEST_F(ArrayMergeTest, MergeAndInterleave3)
{
   vector<array_lifetime> input = {
      {1, 5, 1, 5, WRITEMASK_X},
      {2, 4, 6, 7, WRITEMASK_XY},
      {3, 3, 1, 5, WRITEMASK_X}
   };

   vector<array_remapping> expect = {
      {},
      {1},
      {1, WRITEMASK_X, WRITEMASK_X}
   };
   vector<array_remapping> result(input.size() + 1);
   get_array_remapping(input.size(), &input[0], &result[0]);

   for (unsigned i = 0; i < expect.size(); ++i)
      EXPECT_EQ(result[i + 1], expect[i]);
}

TEST_F(ArrayMergeTest, MergeAndInterleave4)
{
   vector<array_lifetime> input = {
      {1, 7, 1, 5, WRITEMASK_X},
      {2, 6, 6, 7, WRITEMASK_XY},
      {3, 5, 1, 5, WRITEMASK_X},
      {4, 4, 8, 9, WRITEMASK_XYZ},
      {5, 3, 8, 9, WRITEMASK_W},
      {6, 2, 10, 11, WRITEMASK_XYZW},
   };

   vector<array_remapping> expect = {
      {},
      {1},
      {1, WRITEMASK_X, WRITEMASK_X},
      {1},
      {1, WRITEMASK_XYZ, WRITEMASK_W},
      {1}
   };
   vector<array_remapping> result(input.size() + 1);
   get_array_remapping(input.size(), &input[0], &result[0]);

   EXPECT_EQ(result[1], expect[0]);
   EXPECT_EQ(result[2], expect[1]);
   EXPECT_EQ(result[3], expect[2]);
   EXPECT_EQ(result[4], expect[3]);
   EXPECT_EQ(result[5], expect[4]);
   EXPECT_EQ(result[6], expect[5]);

}
