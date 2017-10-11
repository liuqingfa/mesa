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

#include <state_tracker/st_glsl_to_tgsi_temprename.h>
#include <tgsi/tgsi_ureg.h>
#include <tgsi/tgsi_info.h>
#include <compiler/glsl/list.h>
#include <mesa/program/prog_instruction.h>

#include <utility>
#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>

using std::vector;
using std::pair;
using std::make_pair;
using std::transform;
using std::copy;
using std::tuple;

/* Use this to make the compiler pick the swizzle constructor below */
struct SWZ {};

/* Use this to make the compiler pick the constructor with reladdr below */
struct RA {};

/* A line to describe a TGSI instruction for building mock shaders. */
struct MockCodeline {
   MockCodeline(unsigned _op): op(_op), max_temp_id(0){}
   MockCodeline(unsigned _op, const vector<int>& _dst, const vector<int>& _src,
                const vector<int>&_to);

   MockCodeline(unsigned _op, const vector<pair<int,int>>& _dst,
                const vector<pair<int, const char *>>& _src,
                const vector<pair<int, const char *>>&_to, SWZ with_swizzle);

   MockCodeline(unsigned _op, const vector<tuple<int,int,int>>& _dst,
                const vector<tuple<int,int,int>>& _src,
                const vector<tuple<int,int,int>>&_to, RA with_reladdr);

   int get_max_reg_id() const { return max_temp_id;}

   glsl_to_tgsi_instruction *get_codeline() const;

   static void set_mem_ctx(void *ctx);

private:
   st_src_reg create_src_register(int src_idx);
   st_src_reg create_src_register(int src_idx, const char *swizzle);
   st_src_reg create_src_register(int src_idx, gl_register_file file);
   st_src_reg create_src_register(const tuple<int,int,int>& src);
   st_src_reg *create_rel_src_register(int idx);

   st_dst_reg create_dst_register(int dst_idx);
   st_dst_reg create_dst_register(int dst_idx, int writemask);
   st_dst_reg create_dst_register(int dst_idx, gl_register_file file);
   st_dst_reg create_dst_register(const tuple<int,int,int>& dest);
   unsigned op;
   vector<st_dst_reg> dst;
   vector<st_src_reg> src;
   vector<st_src_reg> tex_offsets;

   int max_temp_id;
   static void *mem_ctx;
};

/* A few constants that will notbe tracked as temporary registers by the
 * mock shader.
 */
const int in0 = -1;
const int in1 = -2;
const int in2 = -3;

const int out0 = -1;
const int out1 = -2;

class MockShader {
public:
   MockShader(const vector<MockCodeline>& source, void *ctx);

   exec_list* get_program() const;
   int get_num_temps() const;

private:
   exec_list* program;
   int num_temps;
};

using expectation = vector<vector<int>>;

class MesaTestWithMemCtx : public testing::Test {
   void SetUp();
   void TearDown();
protected:
   void *mem_ctx;
};

class LifetimeEvaluatorTest : public MesaTestWithMemCtx {
protected:
   void run(const vector<MockCodeline>& code, const expectation& e);
private:
   virtual void check(const vector<lifetime>& result, const expectation& e) = 0;
};

/* This is a test class to check the exact life times of
 * registers. */
class LifetimeEvaluatorExactTest : public LifetimeEvaluatorTest {
protected:
   void check(const vector<lifetime>& result, const expectation& e);
};

/* This test class checks that the life time covers at least
 * in the expected range. It is used for cases where we know that
 * a the implementation could be improved on estimating the minimal
 * life time.
 */
class LifetimeEvaluatorAtLeastTest : public LifetimeEvaluatorTest {
protected:
   void check(const vector<lifetime>& result, const expectation& e);
};

/* With this test class the renaming mapping estimation is tested */
class RegisterRemappingTest : public MesaTestWithMemCtx {
protected:
   void run(const vector<lifetime>& lt, const vector<int>& expect);
};

/* With this test class the combined lifetime estimation and renaming
 * mepping estimation is tested
 */
class RegisterLifetimeAndRemappingTest : public RegisterRemappingTest  {
protected:
   using RegisterRemappingTest::run;
   void run(const vector<MockCodeline>& code, const vector<int>& expect);
};

TEST_F(LifetimeEvaluatorExactTest, SimpleMoveAdd)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_UADD, {out0}, {1,in0}, {}},
      { TGSI_OPCODE_END}
   };
   run(code, expectation({{-1,-1}, {0,1}}));
}

TEST_F(LifetimeEvaluatorExactTest, SimpleMoveAddMove)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_UADD, {2}, {1,in0}, {}},
      { TGSI_OPCODE_MOV, {out0}, {2}, {}},
      { TGSI_OPCODE_END}
   };
   run(code, expectation({{-1, -1}, {0,1}, {1,2}}));
}

/* Test whether the texoffst are actually visited by the
 * merge algorithm. Note that it is of no importance
 * what instruction is actually used, the MockShader class
 * does not consider the details of the operation, only
 * the number of arguments is of importance.
 */
TEST_F(LifetimeEvaluatorExactTest, SimpleOpWithTexoffset)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_MOV, {2}, {in1}, {}},
      { TGSI_OPCODE_TEX, {out0}, {in0}, {1,2}},
      { TGSI_OPCODE_END}
   };
   run(code, expectation({{-1, -1}, {0,2}, {1,2}}));
}

/* Simple register access involving a loop
 * 1: must life up to then end of the loop
 * 2: only needs to life from write to read
 * 3: only needs to life from write to read outside the loop
 */
TEST_F(LifetimeEvaluatorExactTest, SimpleMoveInLoop)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_UADD, {2}, {1,in0}, {}},
      {   TGSI_OPCODE_UADD, {3}, {1,2}, {}},
      {   TGSI_OPCODE_UADD, {3}, {3,in1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {3}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,5}, {2,3}, {3,6}}));
}

/* In loop if/else value written only in one path, and read later
 * - value must survive the whole loop.
 */
TEST_F(LifetimeEvaluatorExactTest, MoveInIfInLoop)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in1}, {}},
      {     TGSI_OPCODE_UADD, {2}, {1,in0}, {}},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_UADD, {3}, {1,2}, {}},
      {   TGSI_OPCODE_UADD, {3}, {3,in1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {3}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,7}, {1,7}, {5,8}}));
}

/* A non-dominant write within an IF can be ignored (if it is read
 * later)
 */
TEST_F(LifetimeEvaluatorExactTest, NonDominantWriteinIfInLoop)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {   TGSI_OPCODE_IF, {}, {in1}, {}},
      {     TGSI_OPCODE_MOV, {1}, {in1}, {}},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_UADD, {2}, {1,in1}, {}},
      {   TGSI_OPCODE_IF, {}, {2}, {}},
      {     TGSI_OPCODE_BRK},
      {   TGSI_OPCODE_ENDIF},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {2}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {1,5}, {5,10}}));
}

/* In Nested loop if/else value written only in one path, and read later
 * - value must survive the outer loop.
 */
TEST_F(LifetimeEvaluatorExactTest, MoveInIfInNestedLoop)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_BGNLOOP },
      {     TGSI_OPCODE_IF, {}, {in1}, {} },
      {       TGSI_OPCODE_UADD, {2}, {1,in0}, {}},
      {     TGSI_OPCODE_ENDIF},
      {     TGSI_OPCODE_UADD, {3}, {1,2}, {}},
      {   TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {3}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,8}, {1,8}, {6,9}}));
}

/* In loop if/else value written in both path, and read later
 * - value must survive from first write to last read in loop
 * for now we only check that the minimum life time is correct.
 */
TEST_F(LifetimeEvaluatorExactTest, WriteInIfAndElseInLoop)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {1}, {}},
      {     TGSI_OPCODE_UADD, {2}, {1,in0}, {}},
      {   TGSI_OPCODE_ELSE },
      {     TGSI_OPCODE_MOV, {2}, {1}, {}},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_UADD, {3}, {1,2}, {}},
      {   TGSI_OPCODE_UADD, {3}, {3,in1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {3}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,9}, {3,7}, {7,10}}));
}

/* Test that read before write in ELSE path is properly tracked:
 * In loop if/else value written in both path but read in else path
 * before write and also read later - value must survive the whole loop.
 */
TEST_F(LifetimeEvaluatorExactTest, WriteInIfAndElseReadInElseInLoop)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {1}, {}},
      {     TGSI_OPCODE_UADD, {2}, {1,in0}, {}},
      {   TGSI_OPCODE_ELSE },
      {     TGSI_OPCODE_ADD, {2}, {1,2}, {}},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_UADD, {3}, {1,2}, {}},
      {   TGSI_OPCODE_UADD, {3}, {3,in1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {3}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,9}, {1,9}, {7,10}}));
}


/* Test that a write in ELSE path only in loop is properly tracked:
 * In loop if/else value written in else path and read outside
 * - value must survive the whole loop.
 */
TEST_F(LifetimeEvaluatorExactTest, WriteInElseReadInLoop)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {1}, {}},
      {     TGSI_OPCODE_UADD, {2}, {1,in0}, {}},
      {   TGSI_OPCODE_ELSE },
      {     TGSI_OPCODE_ADD, {3}, {1,2}, {}},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_UADD, {1}, {3,in1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,9}, {1,8}, {1,8}}));
}

/* Test that tracking a second write in an ELSE path is not attributed
 * to the IF path: In loop if/else value written in else path twice and
 * read outside - value must survive the whole loop
 */
TEST_F(LifetimeEvaluatorExactTest, WriteInElseTwiceReadInLoop)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {1}, {}},
      {     TGSI_OPCODE_UADD, {2}, {1,in0}, {}},
      {   TGSI_OPCODE_ELSE },
      {     TGSI_OPCODE_ADD, {3}, {1,2}, {}},
      {     TGSI_OPCODE_ADD, {3}, {1,3}, {}},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_UADD, {1}, {3,in1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,10}, {1,9}, {1,9}}));
}

/* Test that the IF and ELSE scopes from different IF/ELSE pairs are not
 * merged: In loop if/else value written in if, and then in different else path
 * and read outside - value must survive the whole loop
 */
TEST_F(LifetimeEvaluatorExactTest, WriteInOneIfandInAnotherElseInLoop)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {1}, {}},
      {     TGSI_OPCODE_UADD, {2}, {1,in0}, {}},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_IF, {}, {1}, {}},
      {   TGSI_OPCODE_ELSE },
      {     TGSI_OPCODE_ADD, {2}, {1,1}, {}},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_UADD, {1}, {2,in1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,11}, {1,10}}));
}

/* Test that with a new loop the resolution of the IF/ELSE write conditionality
 * is restarted: In first loop value is written in both if and else, in second
 * loop value is written only in if - must survive the second loop.
 * However, the tracking is currently not able to restrict the lifetime
 * in the first loop, hence the "AtLeast" test.
 */
TEST_F(LifetimeEvaluatorAtLeastTest, UnconditionalInFirstLoopConditionalInSecond)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {1}, {}},
      {     TGSI_OPCODE_UADD, {2}, {1,in0}, {}},
      {   TGSI_OPCODE_ELSE },
      {     TGSI_OPCODE_UADD, {2}, {1,in1}, {}},
      {   TGSI_OPCODE_ENDIF},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {1}, {}},
      {     TGSI_OPCODE_ADD, {2}, {in0,1}, {}},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_UADD, {1}, {2,in1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,14}, {3,13}}));
}

/* Test that with a new loop the resolution of the IF/ELSE write conditionality
 * is restarted, and also takes care of write before read in else scope:
 * In first loop value is written in both if and else, in second loop value is
 * also written in both, but first read in if - must survive the second loop.
 * However, the tracking is currently not able to restrict the lifetime
 * in the first loop, hence the "AtLeast" test.
 */
TEST_F(LifetimeEvaluatorAtLeastTest, UnconditionalInFirstLoopConditionalInSecond2)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {1}, {}},
      {     TGSI_OPCODE_UADD, {2}, {1,in0}, {}},
      {   TGSI_OPCODE_ELSE },
      {     TGSI_OPCODE_UADD, {2}, {1,in1}, {}},
      {   TGSI_OPCODE_ENDIF},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in1}, {}},
      {     TGSI_OPCODE_ADD, {2}, {2,1}, {}},
      {   TGSI_OPCODE_ELSE },
      {     TGSI_OPCODE_MOV, {2}, {1}, {}},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_UADD, {1}, {2,in1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,16}, {3,15}}));
}

/* In loop if/else read in one path before written in the same loop
 * - value must survive the whole loop
 */
TEST_F(LifetimeEvaluatorExactTest, ReadInIfInLoopBeforeWrite)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_UADD, {2}, {1,3}, {}},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_UADD, {3}, {1,2}, {}},
      {   TGSI_OPCODE_UADD, {3}, {3,in1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {3}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,7}, {1,7}, {1,8}}));
}

/* In loop if/else read in one path before written in the same loop
 * read after the loop, value must survivethe whole loop and
 * to the read.
 */
TEST_F(LifetimeEvaluatorExactTest, ReadInLoopInIfBeforeWriteAndLifeToTheEnd)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_MUL, {1}, {1,in1}, {}},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_UADD, {1}, {1,in1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,6}}));
}

/* In loop read before written in the same loop read after the loop,
 * value must survive the whole loop and to the read.
 * This is kind of undefined behaviour though ...
 */
TEST_F(LifetimeEvaluatorExactTest, ReadInLoopBeforeWriteAndLifeToTheEnd)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_MUL, {1}, {1,in1}, {}},
      {   TGSI_OPCODE_UADD, {1}, {1,in1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,4}}));
}

/* Test whether nesting IF/ELSE pairs within a loop is resolved:
 * Write in all conditional branches if the inner nesting level and
 * read after the outer IF/ELSE pair is closed. The lifetime doesn't have
 * to be extended to the full loop.
 */
TEST_F(LifetimeEvaluatorExactTest, NestedIfInLoopAlwaysWriteButNotPropagated)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_IF, {}, {in0}, {}},
      {       TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_ELSE},
      {       TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ELSE},
      {     TGSI_OPCODE_IF, {}, {in0}, {}},
      {       TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_ELSE},
      {       TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {3,14}}));
}

/* Test that nested chaining of IF/ELSE scopes is resolved:
 * Write in each IF branch, and open another IF/ELSE scope pair in the ELSE
 * branch. At the last nesting level, the temporary is also written in the
 * ELSE branch, hence the full constrict results in an unconditional write.
 */
TEST_F(LifetimeEvaluatorExactTest, DeeplyNestedIfElseInLoopResolved)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {   TGSI_OPCODE_ELSE},
      {     TGSI_OPCODE_IF, {}, {in0}, {}},
      {       TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_ELSE},
      {       TGSI_OPCODE_IF, {}, {in0}, {}},
      {         TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {       TGSI_OPCODE_ELSE},
      {         TGSI_OPCODE_IF, {}, {in0}, {}},
      {           TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {         TGSI_OPCODE_ELSE},
      {           TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {         TGSI_OPCODE_ENDIF},
      {       TGSI_OPCODE_ENDIF},
      {     TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ADD, {2}, {1, in1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {2}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {2,18}, {18, 20}}));
}

/* The complementary case of the above: Open deeply nested IF/ELSE clauses
 * and only at the deepest nesting level the temporary is written in the IF
 * branch, but for all ELSE scopes the value is also written. Like above, when
 * the full construct has been executed, the temporary has been written
 * unconditionally.
 */
TEST_F(LifetimeEvaluatorExactTest, DeeplyNestedIfElseInLoopResolved2)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_IF, {}, {in0}, {}},
      {       TGSI_OPCODE_IF, {}, {in0}, {}},
      {         TGSI_OPCODE_IF, {}, {in0}, {}},
      {           TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {         TGSI_OPCODE_ELSE},
      {           TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {         TGSI_OPCODE_ENDIF},
      {       TGSI_OPCODE_ELSE},
      {         TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {       TGSI_OPCODE_ENDIF},
      {     TGSI_OPCODE_ELSE},
      {       TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ELSE},
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ADD, {2}, {1, in1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {2}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {5,18}, {18, 20}}));
}

/* Test that a write in an IF scope within IF scope where the temporary already
 * can be ignored.
 */
TEST_F(LifetimeEvaluatorExactTest, NestedIfElseInLoopResolvedInOuterScope)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_IF, {}, {in0}, {}},
      {       TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ELSE},
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ADD, {2}, {1, in1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {2}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {2,9}, {9, 11}}));
}

/* Here the read before write in the nested if is of no consequence to the
 * life time because the variable was already written in the enclosing if-branch.
 */
TEST_F(LifetimeEvaluatorExactTest, NestedIfElseInLoopWithReadResolvedInOuterScope)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_IF, {}, {in0}, {}},
      {       TGSI_OPCODE_ADD, {1}, {in0, 1}, {}},
      {     TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ELSE},
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ADD, {2}, {1, in1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {2}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {2,9}, {9, 11}}));
}

/* Here the nested if condition is of no consequence to the life time
 * because the variable was already written in the enclosing else-branch.
 */
TEST_F(LifetimeEvaluatorExactTest, NestedIfElseInLoopResolvedInOuterScope2)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {   TGSI_OPCODE_ELSE},
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_IF, {}, {in0}, {}},
      {       TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ADD, {2}, {1, in1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {2}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {2,9}, {9, 11}}));
}

/* Test that tracking of IF/ELSE scopes does not unnessesarily cross loops,
 * i.e. if the inner IF/ELSE pair is enclosed by a loop which is enclosed
 * by another IF statement: The resolution of unconditionality of the write
 * within the loop is not changed by the fact that the loop is enclosed by
 * an IF scope.
 */
TEST_F(LifetimeEvaluatorExactTest, NestedIfInLoopAlwaysWriteParentIfOutsideLoop)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_IF, {}, {in0}, {}},
      {   TGSI_OPCODE_BGNLOOP },
      {     TGSI_OPCODE_IF, {}, {in0}, {}},
      {       TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_ELSE},
      {       TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_ENDIF},
      {     TGSI_OPCODE_IF, {}, {in0}, {}},
      {       TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_ELSE},
      {       TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_MOV, {2}, {1}, {}},
      {   TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_ELSE},
      {   TGSI_OPCODE_MOV, {2}, {in1}, {}},
      { TGSI_OPCODE_ENDIF},
      { TGSI_OPCODE_MOV, {out0}, {2}, {}},

      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {3,12}, {12, 17}}));
}

/* The value is written in a loop and in a nested IF, but
 * not in all code paths, hence the value must survive the loop.
 */
TEST_F(LifetimeEvaluatorExactTest, NestedIfInLoopWriteNotAlways)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_IF, {}, {in0}, {}},
      {       TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_ELSE},
      {       TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ELSE},
      {     TGSI_OPCODE_IF, {}, {in0}, {}},
      {       TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,13}}));
}

/* Test that reading in an ELSE branach after writing is ignored:
 * The value is written in a loop in both branches of if-else but also
 * read in the else after writing, should have no effect on lifetime.
 */
TEST_F(LifetimeEvaluatorExactTest, IfElseWriteInLoopAlsoReadInElse)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {   TGSI_OPCODE_ELSE},
      {     TGSI_OPCODE_MOV, {1}, {in1}, {}},
      {     TGSI_OPCODE_MUL, {1}, {in0, 1}, {}},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {2,7}}));
}

/* Test that a write in an inner IF/ELSE pair is propagated to the outer
 * ELSE branch: The value is written in a loop in both branches of a nested
 * IF/ELSE pair, but only within the outer else, hence in summary the write is
 * conditional within the loop.
 */
TEST_F(LifetimeEvaluatorExactTest, WriteInNestedIfElseOuterElseOnly)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {   TGSI_OPCODE_ELSE},
      {     TGSI_OPCODE_IF, {}, {in0}, {}},
      {       TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_ELSE},
      {       TGSI_OPCODE_ADD, {1}, {in1, in0}, {}},
      {     TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,10}}));
}

/* Test that reads in an inner ELSE after write within the enclosing IF branch
 * is of no consequence (i.e. check that the read in the ELSE branch is not
 * attributed as read before write when the outer ELSE branch is scanned:
 * Nested if-else in loop. The value is written in the outer if and else and
 * read in one inner else, should limit lifetime.
 */
TEST_F(LifetimeEvaluatorExactTest, WriteUnconditionallyReadInNestedElse)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_ELSE},
      {       TGSI_OPCODE_MOV, {out1}, {1}, {}},
      {     TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ELSE},
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {2,10}}));
}


/* Nested if-else in loop. The value is written in a loop in both branches
 * of if-else but also read in the second nested else before writing.
 * Is conditional.
 */
TEST_F(LifetimeEvaluatorExactTest, NestedIfelseReadFirstInInnerElseInLoop)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_IF, {}, {in0}, {}},
      {       TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_ELSE},
      {       TGSI_OPCODE_MOV, {1}, {in1}, {}},
      {     TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ELSE},
      {     TGSI_OPCODE_IF, {}, {in0}, {}},
      {       TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_ELSE},
      {       TGSI_OPCODE_ADD, {1}, {in1, 1}, {}},
      {     TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,15}}));
}

/* Test that read before write is properly tracked for nested IF branches.
 * The value is written in a loop in both branches of IF/ELSE but also read in
 * the second nested IF before writing - is conditional.
 */
TEST_F(LifetimeEvaluatorExactTest, NestedIfelseReadFirstInInnerIfInLoop)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_IF, {}, {in0}, {}},
      {       TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_ELSE},
      {       TGSI_OPCODE_MOV, {1}, {in1}, {}},
      {     TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ELSE},
      {     TGSI_OPCODE_IF, {}, {in0}, {}},
      {       TGSI_OPCODE_ADD, {1}, {in1, 1}, {}},
      {     TGSI_OPCODE_ELSE},
      {       TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,15}}));
}

/* Same as above, but for the secondary ELSE branch:
 * The value is written in a loop in both branches of IF/ELSE but also read in
 * the second nested ELSE branch before writing - is conditional.
 */
TEST_F(LifetimeEvaluatorExactTest, WriteInOneElseBranchReadFirstInOtherInLoop)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {   TGSI_OPCODE_ELSE},
      {     TGSI_OPCODE_MOV, {1}, {in1}, {}},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {   TGSI_OPCODE_ELSE},
      {     TGSI_OPCODE_ADD, {1}, {in1, 1}, {}},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {2,11}}));
}

/* Test that the "write is unconditional" resolution is not overwritten within
 * a loop: The value is written in a loop in both branches of an IF/ELSE clause,
 * hence the second IF doesn't make it conditional.
 */
TEST_F(LifetimeEvaluatorExactTest, WriteInIfElseBranchSecondIfInLoop)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {   TGSI_OPCODE_ELSE},
      {     TGSI_OPCODE_MOV, {1}, {in1}, {}},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {2,9}}));
}


/* A continue in the loop is not relevant */
TEST_F(LifetimeEvaluatorExactTest, LoopWithWriteAfterContinue)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_CONT},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {4,6}}));
}

/* Temporary used to in case must live up to the case
 * statement where it is used, the switch we only keep
 * for the actual SWITCH opcode like it is in tgsi_exec.c, the
 * only current use case.
 */
TEST_F(LifetimeEvaluatorExactTest, UseSwitchCase)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_MOV, {2}, {in1}, {}},
      { TGSI_OPCODE_MOV, {3}, {in2}, {}},
      { TGSI_OPCODE_SWITCH, {}, {3}, {}},
      {   TGSI_OPCODE_CASE, {}, {2}, {}},
      {   TGSI_OPCODE_CASE, {}, {1}, {}},
      {   TGSI_OPCODE_BRK},
      {   TGSI_OPCODE_DEFAULT},
      { TGSI_OPCODE_ENDSWITCH},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,5}, {1,4}, {2,3}}));
}

/* With two destinations, if one result is thrown away, the
 * register must be kept past the writing instructions.
 */
TEST_F(LifetimeEvaluatorExactTest, WriteTwoOnlyUseOne)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_DFRACEXP , {1,2}, {in0}, {}},
      { TGSI_OPCODE_ADD , {3}, {2,in0}, {}},
      { TGSI_OPCODE_MOV, {out1}, {3}, {}},
      { TGSI_OPCODE_END},

   };
   run (code, expectation({{-1,-1}, {0,1}, {0,1}, {1,2}}));
}

/* If a break is in the loop, all variables written after the
 * break and used outside the loop must be maintained for the
 * whole loop
 */
TEST_F(LifetimeEvaluatorExactTest, LoopWithWriteAfterBreak)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_BRK},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,6}}));
}

/* If a break is in the loop, all variables written after the
 * break and used outside the loop must be maintained for the
 * whole loop. The first break in the loop is the defining one.
 */
TEST_F(LifetimeEvaluatorExactTest, LoopWithWriteAfterBreak2Breaks)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_BRK},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {   TGSI_OPCODE_BRK},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,7}}));
}

/* Loop with a break at the beginning and read/write in the post
 * break loop scope. The value written and read within the loop
 * can be limited to [write, read], but the value read outside the
 * loop must survive the whole loop. This is the typical code for
 * while and for loops, where the breaking condition is tested at
 * the beginning.
 */
TEST_F(LifetimeEvaluatorExactTest, LoopWithWriteAndReadAfterBreak)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_BRK},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {   TGSI_OPCODE_MOV, {2}, {1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {2}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {4,5}, {0,7}}));
}

/* Same as above, just make sure that the life time of the local variable
 * in the outer loop (3) is not accidently promoted to the whole loop.
 */
TEST_F(LifetimeEvaluatorExactTest, NestedLoopWithWriteAndReadAfterBreak)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_IF, {}, {in1}, {}},
      {     TGSI_OPCODE_BRK},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_BGNLOOP},
      {     TGSI_OPCODE_IF, {}, {in0}, {}},
      {       TGSI_OPCODE_BRK},
      {     TGSI_OPCODE_ENDIF},
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_MOV, {2}, {1}, {}},
      {   TGSI_OPCODE_ENDLOOP },
      {   TGSI_OPCODE_ADD, {3}, {2,in0}, {}},
      {   TGSI_OPCODE_ADD, {4}, {3,in2}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {4}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {8,9}, {0,13}, {11,12}, {0,14}}));
}

/* If a break is in the loop inside a switch case, make sure it is
 * interpreted as breaking that inner loop, i.e. the variable has to
 * survive the loop.
 */
TEST_F(LifetimeEvaluatorExactTest, LoopWithWriteAfterBreakInSwitchInLoop)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_SWITCH, {}, {in1}, {}},
      {  TGSI_OPCODE_CASE, {}, {in1}, {}},
      {   TGSI_OPCODE_BGNLOOP },
      {    TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_BRK},
      {    TGSI_OPCODE_ENDIF},
      {    TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {   TGSI_OPCODE_ENDLOOP },
      {  TGSI_OPCODE_DEFAULT, {}, {}, {}},
      { TGSI_OPCODE_ENDSWITCH, {}, {}, {}},
      { TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {2,10}}));
}

/* Value written conditionally in one loop and read in another loop,
 * and both of these loops are within yet another loop. Here the value
 * has to survive the outer loop.
 */
TEST_F(LifetimeEvaluatorExactTest, LoopsWithDifferntScopesConditionalWrite)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {  TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {   TGSI_OPCODE_ENDIF},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_BGNLOOP },
      {     TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,7}}));
}

/* Value written and read in one loop and last read in another loop,
 * Here the value has to survive both loops.
 */
TEST_F(LifetimeEvaluatorExactTest, LoopsWithDifferntScopesFirstReadBeforeWrite)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_MUL, {1}, {1,in0}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,5}}));
}


/* Value is written in one switch code path within a loop
 * must survive the full loop.
 */
TEST_F(LifetimeEvaluatorExactTest, LoopWithWriteInSwitch)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_SWITCH, {}, {in0}, {} },
      {    TGSI_OPCODE_CASE, {}, {in0}, {} },
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {    TGSI_OPCODE_BRK },
      {   TGSI_OPCODE_DEFAULT },
      {   TGSI_OPCODE_BRK },
      {   TGSI_OPCODE_ENDSWITCH },
      {   TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,9}}));
}

/* Value written in one case, and read in other,in loop
 * - must survive the loop.
 */
TEST_F(LifetimeEvaluatorExactTest, LoopWithReadWriteInSwitchDifferentCase)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_SWITCH, {}, {in0}, {} },
      {    TGSI_OPCODE_CASE, {}, {in0}, {} },
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {    TGSI_OPCODE_BRK },
      {   TGSI_OPCODE_DEFAULT },
      {     TGSI_OPCODE_MOV, {out0}, {1}, {}},
      {   TGSI_OPCODE_BRK },
      {   TGSI_OPCODE_ENDSWITCH },
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,9}}));
}

/* Value written in one case, and read in other,in loop
 * - must survive the loop, even if the write case falls through.
 */
TEST_F(LifetimeEvaluatorExactTest, LoopWithReadWriteInSwitchDifferentCaseFallThrough)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_SWITCH, {}, {in0}, {} },
      {    TGSI_OPCODE_CASE, {}, {in0}, {} },
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {   TGSI_OPCODE_DEFAULT },
      {     TGSI_OPCODE_MOV, {out0}, {1}, {}},
      {   TGSI_OPCODE_BRK },
      {   TGSI_OPCODE_ENDSWITCH },
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,8}}));
}

/* Here we read and write from an to the same temp in the same instruction,
 * but the read is conditional (select operation), hence the lifetime must
 * start with the first write.
 */
TEST_F(LifetimeEvaluatorExactTest, WriteSelectFromSelf)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_USEQ, {5}, {in0,in1}, {}},
      { TGSI_OPCODE_UCMP, {1}, {5,in1,1}, {}},
      { TGSI_OPCODE_UCMP, {1}, {5,in1,1}, {}},
      { TGSI_OPCODE_UCMP, {1}, {5,in1,1}, {}},
      { TGSI_OPCODE_UCMP, {1}, {5,in1,1}, {}},
      { TGSI_OPCODE_FSLT, {2}, {1,in1}, {}},
      { TGSI_OPCODE_UIF, {}, {2}, {}},
      {   TGSI_OPCODE_MOV, {3}, {in1}, {}},
      { TGSI_OPCODE_ELSE},
      {   TGSI_OPCODE_MOV, {4}, {in1}, {}},
      {   TGSI_OPCODE_MOV, {4}, {4}, {}},
      {   TGSI_OPCODE_MOV, {3}, {4}, {}},
      { TGSI_OPCODE_ENDIF},
      { TGSI_OPCODE_MOV, {out1}, {3}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {1,5}, {5,6}, {7,13}, {9,11}, {0,4}}));
}

/* This test checks wheter the ENDSWITCH is handled properly if the
 * last switch case/default doesn't stop with a BRK.
 */
TEST_F(LifetimeEvaluatorExactTest, LoopRWInSwitchCaseLastCaseWithoutBreak)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_SWITCH, {}, {in0}, {} },
      {    TGSI_OPCODE_CASE, {}, {in0}, {} },
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {    TGSI_OPCODE_BRK },
      {   TGSI_OPCODE_DEFAULT },
      {     TGSI_OPCODE_MOV, {out0}, {1}, {}},
      {   TGSI_OPCODE_ENDSWITCH },
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,8}}));
}

/* Value read/write in same case, stays there */
TEST_F(LifetimeEvaluatorExactTest, LoopWithReadWriteInSwitchSameCase)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_SWITCH, {}, {in0}, {} },
      {    TGSI_OPCODE_CASE, {}, {in0}, {} },
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {     TGSI_OPCODE_MOV, {out0}, {1}, {}},
      {    TGSI_OPCODE_BRK },
      {   TGSI_OPCODE_DEFAULT },
      {   TGSI_OPCODE_BRK },
      {   TGSI_OPCODE_ENDSWITCH },
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {3,4}}));
}

/* Value read/write in all cases, should only live from first
 * write to last read, but currently the whole loop is used.
 */
TEST_F(LifetimeEvaluatorAtLeastTest, LoopWithReadWriteInSwitchSameCase)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_SWITCH, {}, {in0}, {}},
      {    TGSI_OPCODE_CASE, {}, {in0}, {} },
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {    TGSI_OPCODE_BRK },
      {   TGSI_OPCODE_DEFAULT },
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {   TGSI_OPCODE_BRK },
      {   TGSI_OPCODE_ENDSWITCH },
      {   TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {3,9}}));
}

/* First read before first write with nested loops */
TEST_F(LifetimeEvaluatorExactTest, LoopsWithDifferentScopesCondReadBeforeWrite)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_BGNLOOP },
      {    TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_MOV, {out0}, {1}, {}},
      {    TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ENDLOOP },
      {   TGSI_OPCODE_BGNLOOP },
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {   TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,9}}));
}

/* First read before first write wiredness with nested loops.
 * Here the first read of 2 is logically before the first, dominant
 * write, therfore, the 2 has to survive both loops.
 */
TEST_F(LifetimeEvaluatorExactTest, FirstWriteAtferReadInNestedLoop)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_BGNLOOP },
      {     TGSI_OPCODE_MUL, {2}, {2,1}, {}},
      {     TGSI_OPCODE_MOV, {3}, {2}, {}},
      {   TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_ADD, {1}, {1,in1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_MOV, {out0}, {3}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,7}, {1,7}, {4,8}}));
}


#define DST(X, W) vector<pair<int,int>>(1, make_pair(X, W))
#define SRC(X, S) vector<pair<int, const char *>>(1, make_pair(X, S))
#define SRC2(X, S, Y, T) vector<pair<int, const char *>>({make_pair(X, S), make_pair(Y, T)})

/* Partial write to components: one component was written unconditionally
 * but another conditionally, temporary must survive the whole loop.
 * Test series for all components.
 */
TEST_F(LifetimeEvaluatorExactTest, LoopWithConditionalComponentWrite_X)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP},
      {   TGSI_OPCODE_MOV, DST(1, WRITEMASK_Y), SRC(in1, "x"), {}, SWZ()},
      {   TGSI_OPCODE_IF, {}, SRC(in0, "xxxx"), {}, SWZ()},
      {     TGSI_OPCODE_MOV, DST(1, WRITEMASK_X), SRC(in1, "y"), {}, SWZ()},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_MOV, DST(2, WRITEMASK_XY), SRC(1, "xy"), {}, SWZ()},
      { TGSI_OPCODE_ENDLOOP},
      { TGSI_OPCODE_MOV, DST(out0, WRITEMASK_XYZW), SRC(2, "xyxy"), {}, SWZ()},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,6}, {5,7}}));
}

TEST_F(LifetimeEvaluatorExactTest, LoopWithConditionalComponentWrite_Y)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP},
      {   TGSI_OPCODE_MOV, DST(1, WRITEMASK_X), SRC(in1, "x"), {}, SWZ()},
      {   TGSI_OPCODE_IF, {}, SRC(in0, "xxxx"), {}, SWZ()},
      {     TGSI_OPCODE_MOV, DST(1, WRITEMASK_Y), SRC(in1, "y"), {}, SWZ()},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_MOV, DST(2, WRITEMASK_XY), SRC(1, "xy"), {}, SWZ()},
      { TGSI_OPCODE_ENDLOOP},
      { TGSI_OPCODE_MOV, DST(out0, WRITEMASK_XYZW), SRC(2, "xyxy"), {}, SWZ()},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,6}, {5,7}}));
}

TEST_F(LifetimeEvaluatorExactTest, LoopWithConditionalComponentWrite_Z)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP},
      {   TGSI_OPCODE_MOV, DST(1, WRITEMASK_X), SRC(in1, "x"), {}, SWZ()},
      {   TGSI_OPCODE_IF, {}, SRC(in0, "xxxx"), {}, SWZ()},
      {     TGSI_OPCODE_MOV, DST(1, WRITEMASK_Z), SRC(in1, "y"), {}, SWZ()},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_MOV, DST(2, WRITEMASK_XY), SRC(1, "xz"), {}, SWZ()},
      { TGSI_OPCODE_ENDLOOP},
      { TGSI_OPCODE_MOV, DST(out0, WRITEMASK_XYZW), SRC(2, "xyxy"), {}, SWZ()},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,6}, {5,7}}));
}

TEST_F(LifetimeEvaluatorExactTest, LoopWithConditionalComponentWrite_W)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP},
      {   TGSI_OPCODE_MOV, DST(1, WRITEMASK_X), SRC(in1, "x"), {}, SWZ()},
      {   TGSI_OPCODE_IF, {}, SRC(in0, "xxxx"), {}, SWZ()},
      {     TGSI_OPCODE_MOV, DST(1, WRITEMASK_W), SRC(in1, "y"), {}, SWZ()},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_MOV, DST(2, WRITEMASK_XY), SRC(1, "xw"), {}, SWZ()},
      { TGSI_OPCODE_ENDLOOP},
      { TGSI_OPCODE_MOV, DST(out0, WRITEMASK_XYZW), SRC(2, "xyxy"), {}, SWZ()},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,6}, {5,7}}));
}

TEST_F(LifetimeEvaluatorExactTest, LoopWithConditionalComponentWrite_X_Read_Y_Before)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP},
      {   TGSI_OPCODE_MOV, DST(1, WRITEMASK_X), SRC(in1, "x"), {}, SWZ()},
      {   TGSI_OPCODE_IF, {}, SRC(in0, "xxxx"), {}, SWZ()},
      {     TGSI_OPCODE_MOV, DST(2, WRITEMASK_XYZW), SRC(1, "yyyy"), {}, SWZ()},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_MOV, DST(1, WRITEMASK_YZW), SRC(2, "yyzw"), {}, SWZ()},
      { TGSI_OPCODE_ENDLOOP},
      { TGSI_OPCODE_ADD, DST(out0, WRITEMASK_XYZW),
                         SRC2(2, "yyzw", 1, "xyxy"), {}, SWZ()},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,7}, {0,7}}));
}

/* The variable is conditionally read before first written, so
 * it has to surive all the loops.
 */
TEST_F(LifetimeEvaluatorExactTest, FRaWSameInstructionInLoopAndCondition)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_BGNLOOP },
      {     TGSI_OPCODE_IF, {}, {in0}, {} },
      {       TGSI_OPCODE_ADD, {1}, {1,in0}, {}},
      {     TGSI_OPCODE_ENDIF},
      {     TGSI_OPCODE_MOV, {1}, {in1}, {}},
      {  TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_END},

   };
   run (code, expectation({{-1,-1}, {0,7}}));
}

/* If unconditionally first written and read in the same
 * instruction, then the register must be kept for the
 * one write, but not more (undefined behaviour)
 */
TEST_F(LifetimeEvaluatorExactTest, FRaWSameInstruction)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_ADD, {1}, {1,in0}, {}},
      { TGSI_OPCODE_END},

   };
   run (code, expectation({{-1,-1}, {0,1}}));
}

/* If unconditionally written and read in the same
 * instruction, various times then the register must be
 * kept past the last write, but not longer (undefined behaviour)
 */
TEST_F(LifetimeEvaluatorExactTest, FRaWSameInstructionMoreThenOnce)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_ADD, {1}, {1,in0}, {}},
      { TGSI_OPCODE_ADD, {1}, {1,in0}, {}},
      { TGSI_OPCODE_MOV, {out0}, {in0}, {}},
      { TGSI_OPCODE_END},

   };
   run (code, expectation({{-1,-1}, {0,2}}));
}

/* Register is only written. This should not happen,
 * but to handle the case we want the register to life
 * at least one instruction
 */
TEST_F(LifetimeEvaluatorExactTest, WriteOnly)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,1}}));
}

/* Register is read in IF.
 */
TEST_F(LifetimeEvaluatorExactTest, SimpleReadForIf)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_ADD, {out0}, {in0,in1}, {}},
      { TGSI_OPCODE_IF, {}, {1}, {}},
      { TGSI_OPCODE_ENDIF}
   };
   run (code, expectation({{-1,-1}, {0,2}}));
}

TEST_F(LifetimeEvaluatorExactTest, WriteTwoReadOne)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_DFRACEXP , {1,2}, {in0}, {}},
      { TGSI_OPCODE_ADD , {3}, {2,in0}, {}},
      { TGSI_OPCODE_MOV, {out1}, {3}, {}},
      { TGSI_OPCODE_END},
   };
   run (code, expectation({{-1,-1}, {0,1}, {0,1}, {1,2}}));
}

TEST_F(LifetimeEvaluatorExactTest, ReadOnly)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_END},
   };
   run (code, expectation({{-1,-1}, {-1,-1}}));
}

/* Test handling of missing END marker
*/
TEST_F(LifetimeEvaluatorExactTest, SomeScopesAndNoEndProgramId)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_IF, {}, {1}, {}},
      { TGSI_OPCODE_MOV, {2}, {1}, {}},
      { TGSI_OPCODE_ENDIF},
      { TGSI_OPCODE_IF, {}, {1}, {}},
      { TGSI_OPCODE_MOV, {out0}, {2}, {}},
      { TGSI_OPCODE_ENDIF},
   };
   run (code, expectation({{-1,-1}, {0,4}, {2,5}}));
}

TEST_F(LifetimeEvaluatorExactTest, SerialReadWrite)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_MOV, {2}, {1}, {}},
      { TGSI_OPCODE_MOV, {3}, {2}, {}},
      { TGSI_OPCODE_MOV, {out0}, {3}, {}},
      { TGSI_OPCODE_END},
   };
   run (code, expectation({{-1,-1}, {0,1}, {1,2}, {2,3}}));
}

/* Check that two destination registers are used */
TEST_F(LifetimeEvaluatorExactTest, TwoDestRegisters)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_DFRACEXP , {1,2}, {in0}, {}},
      { TGSI_OPCODE_ADD, {out0}, {1,2}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,1}, {0,1}}));
}

/* Check that writing within a loop in a conditional is propagated
 * to the outer loop.
 */
TEST_F(LifetimeEvaluatorExactTest, WriteInLoopInConditionalReadOutside)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP},
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_BGNLOOP},
      {       TGSI_OPCODE_MOV, {1}, {in1}, {}},
      {     TGSI_OPCODE_ENDLOOP},
      {   TGSI_OPCODE_ENDIF},
      {   TGSI_OPCODE_ADD, {2}, {1,in1}, {}},
      { TGSI_OPCODE_ENDLOOP},
      { TGSI_OPCODE_MOV, {out0}, {2}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,7}, {6,8}}));
}

/* Check that a register written in a loop that is inside a conditional
 * is not propagated past that loop if last read is also within the
 * conditional
*/
TEST_F(LifetimeEvaluatorExactTest, WriteInLoopInCondReadInCondOutsideLoop)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP},
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_BGNLOOP},
      {       TGSI_OPCODE_MUL, {1}, {in2,in1}, {}},
      {     TGSI_OPCODE_ENDLOOP},
      {     TGSI_OPCODE_ADD, {2}, {1,in1}, {}},
      {   TGSI_OPCODE_ENDIF},
      { TGSI_OPCODE_ENDLOOP},
      { TGSI_OPCODE_MOV, {out0}, {2}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {3,5}, {0,8}}));
}

/* Check that a register read before written in a loop that is
 * inside a conditional is propagated to the outer loop.
 */
TEST_F(LifetimeEvaluatorExactTest, ReadWriteInLoopInCondReadInCondOutsideLoop)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP},
      {   TGSI_OPCODE_IF, {}, {in0}, {}},
      {     TGSI_OPCODE_BGNLOOP},
      {       TGSI_OPCODE_MUL, {1}, {1,in1}, {}},
      {     TGSI_OPCODE_ENDLOOP},
      {     TGSI_OPCODE_ADD, {2}, {1,in1}, {}},
      {   TGSI_OPCODE_ENDIF},
      { TGSI_OPCODE_ENDLOOP},
      { TGSI_OPCODE_MOV, {out0}, {2}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,7}, {0,8}}));
}

/* With two destinations if one value is thrown away, we must
 * ensure that the two output registers don't merge. In this test
 * case the last access for 2 and 3 is in line 4, but 4 can only
 * be merged with 3 because it is read,2 on the other hand is written
 * to, and merging it with 4 would result in a bug.
 */
TEST_F(LifetimeEvaluatorExactTest, WritePastLastRead2)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_MOV, {2}, {in0}, {}},
      { TGSI_OPCODE_ADD, {3}, {1,2}, {}},
      { TGSI_OPCODE_DFRACEXP , {2,4}, {3}, {}},
      { TGSI_OPCODE_MOV, {out1}, {4}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,2}, {1,4}, {2,3}, {3,4}}));
}

/* Check that three source registers are used */
TEST_F(LifetimeEvaluatorExactTest, ThreeSourceRegisters)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_DFRACEXP , {1,2}, {in0}, {}},
      { TGSI_OPCODE_ADD , {3}, {in0,in1}, {}},
      { TGSI_OPCODE_MAD, {out0}, {1,2,3}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,2}, {0,2}, {1,2}}));
}

/* Check minimal lifetime for registers only written to */
TEST_F(LifetimeEvaluatorExactTest, OverwriteWrittenOnlyTemps)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV , {1}, {in0}, {}},
      { TGSI_OPCODE_MOV , {2}, {in1}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,1}, {1,2}}));
}

/* Same register is only written twice. This should not happen,
 * but to handle the case we want the register to life
 * at least past the last write instruction
 */
TEST_F(LifetimeEvaluatorExactTest, WriteOnlyTwiceSame)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,2}}));
}

/* Dead code elimination should catch and remove the case
 * when a variable is written after its last read, but
 * we want the code to be aware of this case.
 * The life time of this uselessly written variable is set
 * to the instruction after the write, because
 * otherwise it could be re-used too early.
 */
TEST_F(LifetimeEvaluatorExactTest, WritePastLastRead)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in0}, {}},
      { TGSI_OPCODE_MOV, {2}, {1}, {}},
      { TGSI_OPCODE_MOV, {1}, {2}, {}},
      { TGSI_OPCODE_END},

   };
   run (code, expectation({{-1,-1}, {0,3}, {1,2}}));
}

/* If a break is in the loop, all variables written after the
 * break and used outside the loop the variable must survive the
 * outer loop
 */
TEST_F(LifetimeEvaluatorExactTest, NestedLoopWithWriteAfterBreak)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_BGNLOOP },
      {   TGSI_OPCODE_BGNLOOP },
      {     TGSI_OPCODE_IF, {}, {in0}, {}},
      {       TGSI_OPCODE_BRK},
      {     TGSI_OPCODE_ENDIF},
      {     TGSI_OPCODE_MOV, {1}, {in0}, {}},
      {   TGSI_OPCODE_ENDLOOP },
      {   TGSI_OPCODE_MOV, {out0}, {1}, {}},
      { TGSI_OPCODE_ENDLOOP },
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,8}}));
}

/* Check lifetime estimation with a relative addressing in src.
 * Note, since the lifetime estimation always extends the lifetime
 * at to at least one instruction after the last write, for the
 * test the last read must be at least two instructions after the
 * last write to obtain a proper test.
 */

TEST_F(LifetimeEvaluatorExactTest, ReadIndirectReladdr1)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV, {1}, {in1}, {}},
      { TGSI_OPCODE_MOV, {2}, {in0}, {}},
      { TGSI_OPCODE_MOV, {{3,0,0}}, {{2,1,0}}, {}, RA()},
      { TGSI_OPCODE_MOV, {out0}, {3}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,2}, {1,2}, {2,3}}));
}

/* Check lifetime estimation with a relative addressing in src */
TEST_F(LifetimeEvaluatorExactTest, ReadIndirectReladdr2)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV , {1}, {in1}, {}},
      { TGSI_OPCODE_MOV , {2}, {in0}, {}},
      { TGSI_OPCODE_MOV , {{3,0,0}}, {{4,0,1}}, {}, RA()},
      { TGSI_OPCODE_MOV , {out0}, {3}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,2}, {1,2},{2,3}}));
}

/* Check lifetime estimation with a relative addressing in src */
TEST_F(LifetimeEvaluatorExactTest, ReadIndirectTexOffsReladdr1)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV , {1}, {in1}, {}},
      { TGSI_OPCODE_MOV , {2}, {in0}, {}},
      { TGSI_OPCODE_MOV , {{3,0,0}}, {{in2,0,0}}, {{5,1,0}}, RA()},
      { TGSI_OPCODE_MOV , {out0}, {3}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,2}, {1,2}, {2,3}}));
}

/* Check lifetime estimation with a relative addressing in src */
TEST_F(LifetimeEvaluatorExactTest, ReadIndirectTexOffsReladdr2)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV , {1}, {in1}, {}},
      { TGSI_OPCODE_MOV , {2}, {in0}, {}},
      { TGSI_OPCODE_MOV , {{3,0,0}}, {{in2,0,0}}, {{2,0,1}}, RA()},
      { TGSI_OPCODE_MOV , {out0}, {3}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,2}, {1,2}, {2,3}}));
}

/* Check lifetime estimation with a relative addressing in dst */
TEST_F(LifetimeEvaluatorExactTest, WriteIndirectReladdr1)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV , {1}, {in0}, {}},
      { TGSI_OPCODE_MOV , {1}, {in1}, {}},
      { TGSI_OPCODE_MOV , {{5,1,0}}, {{in1,0,0}}, {}, RA()},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,2}}));
}

/* Check lifetime estimation with a relative addressing in dst */
TEST_F(LifetimeEvaluatorExactTest, WriteIndirectReladdr2)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_MOV , {1}, {in0}, {}},
      { TGSI_OPCODE_MOV , {2}, {in1}, {}},
      { TGSI_OPCODE_MOV , {{5,0,1}}, {{in1,0,0}}, {}, RA()},
      { TGSI_OPCODE_MOV , {out0}, {in0}, {}},
      { TGSI_OPCODE_MOV , {out1}, {2}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, expectation({{-1,-1}, {0,2}, {1,4}}));
}

/* Test remapping table of registers. The tests don't assume
 * that the sorting algorithm used to sort the lifetimes
 * based on their 'begin' is stable.
 */
TEST_F(RegisterRemappingTest, RegisterRemapping1)
{
   vector<lifetime> lt({{-1,-1},
                        {0,1},
                        {0,2},
                        {1,2},
                        {2,10},
                        {3,5},
                        {5,10}
                       });

   vector<int> expect({0,1,2,1,1,2,2});
   run(lt, expect);
}

TEST_F(RegisterRemappingTest, RegisterRemapping2)
{
   vector<lifetime> lt({{-1,-1},
                        {0,1},
                        {0,2},
                        {3,4},
                        {4,5},
                       });
   vector<int> expect({0,1,2,1,1});
   run(lt, expect);
}

TEST_F(RegisterRemappingTest, RegisterRemappingMergeAllToOne)
{
   vector<lifetime> lt({{-1,-1},
                        {0,1},
                        {1,2},
                        {2,3},
                        {3,4},
                       });
   vector<int> expect({0,1,1,1,1});
   run(lt, expect);
}

TEST_F(RegisterRemappingTest, RegisterRemappingIgnoreUnused)
{
   vector<lifetime> lt({{-1,-1},
                        {0,1},
                        {1,2},
                        {2,3},
                        {-1,-1},
                        {3,4},
                       });
   vector<int> expect({0,1,1,1,4,1});
   run(lt, expect);
}

TEST_F(RegisterRemappingTest, RegisterRemappingMergeZeroLifetimeRegisters)
{
   vector<lifetime> lt({{-1,-1},
                        {0,1},
                        {1,2},
                        {2,3},
                        {3,3},
                        {3,4},
                       });
   vector<int> expect({0,1,1,1,1,1});
   run(lt, expect);
}

TEST_F(RegisterLifetimeAndRemappingTest, LifetimeAndRemapping)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_USEQ, {5}, {in0,in1}, {}},
      { TGSI_OPCODE_UCMP, {1}, {5,in1,1}, {}},
      { TGSI_OPCODE_UCMP, {1}, {5,in1,1}, {}},
      { TGSI_OPCODE_UCMP, {1}, {5,in1,1}, {}},
      { TGSI_OPCODE_UCMP, {1}, {5,in1,1}, {}},
      { TGSI_OPCODE_FSLT, {2}, {1,in1}, {}},
      { TGSI_OPCODE_UIF, {}, {2}, {}},
      {   TGSI_OPCODE_MOV, {3}, {in1}, {}},
      { TGSI_OPCODE_ELSE},
      {   TGSI_OPCODE_MOV, {4}, {in1}, {}},
      {   TGSI_OPCODE_MOV, {4}, {4}, {}},
      {   TGSI_OPCODE_MOV, {3}, {4}, {}},
      { TGSI_OPCODE_ENDIF},
      { TGSI_OPCODE_MOV, {out1}, {3}, {}},
      { TGSI_OPCODE_END}
   };
   run (code, vector<int>({0,1,5,5,1,5}));
}

TEST_F(RegisterLifetimeAndRemappingTest, LifetimeAndRemappingWithUnusedReadOnlyIgnored)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_USEQ, {1}, {in0,in1}, {}},
      { TGSI_OPCODE_UCMP, {2}, {1,in1,2}, {}},
      { TGSI_OPCODE_UCMP, {4}, {2,in1,1}, {}},
      { TGSI_OPCODE_ADD, {5}, {2,4}, {}},
      { TGSI_OPCODE_UIF, {}, {7}, {}},
      {   TGSI_OPCODE_ADD, {8}, {5,4}, {}},
      { TGSI_OPCODE_ENDIF},
      { TGSI_OPCODE_MOV, {out1}, {8}, {}},
      { TGSI_OPCODE_END}
   };
   /* lt: 1: 0-2,2: 1-3 3: u 4: 2-5 5: 3-5 6: u 7: 0-(-1),8: 5-7 */
   run (code, vector<int>({0,1,2,3,1,2,6,7,1}));
}

TEST_F(RegisterLifetimeAndRemappingTest, LifetimeAndRemappingWithUnusedReadOnlyRemappedTo)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_USEQ, {1}, {in0,in1}, {}},
      { TGSI_OPCODE_UIF, {}, {7}, {}},
      {   TGSI_OPCODE_UCMP, {2}, {1,in1,2}, {}},
      {   TGSI_OPCODE_UCMP, {4}, {2,in1,1}, {}},
      {   TGSI_OPCODE_ADD, {5}, {2,4}, {}},
      {   TGSI_OPCODE_ADD, {8}, {5,4}, {}},
      { TGSI_OPCODE_ENDIF},
      { TGSI_OPCODE_MOV, {out1}, {8}, {}},
      { TGSI_OPCODE_END}
   };
   /* lt: 1: 0-3,2: 2-4 3: u 4: 3-5 5: 4-5 6: u 7: 1-1,8: 5-7 */
   run (code, vector<int>({0,1,2,3,1,2,6,7,1}));
}

TEST_F(RegisterLifetimeAndRemappingTest, LifetimeAndRemappingWithUnusedReadOnlyRemapped)
{
   const vector<MockCodeline> code = {
      { TGSI_OPCODE_USEQ, {0}, {in0,in1}, {}},
      { TGSI_OPCODE_UCMP, {2}, {0,in1,2}, {}},
      { TGSI_OPCODE_UCMP, {4}, {2,in1,0}, {}},
      { TGSI_OPCODE_UIF, {}, {7}, {}},
      {   TGSI_OPCODE_ADD, {5}, {4,4}, {}},
      {   TGSI_OPCODE_ADD, {8}, {5,4}, {}},
      { TGSI_OPCODE_ENDIF},
      { TGSI_OPCODE_MOV, {out1}, {8}, {}},
      { TGSI_OPCODE_END}
   };
   /* lt: 0: 0-2 1: u 2: 1-2 3: u 4: 2-5 5: 4-5 6: u 7:ro 8: 5-7 */
   run (code, vector<int>({0,1,2,3,0,2,6,7,0}));
}

/* Implementation of helper and test classes */
void *MockCodeline::mem_ctx = nullptr;

MockCodeline::MockCodeline(unsigned _op, const vector<int>& _dst,
                           const vector<int>& _src, const vector<int>&_to):
   op(_op),
   max_temp_id(0)
{
   transform(_dst.begin(), _dst.end(), std::back_inserter(dst),
             [this](int i) { return create_dst_register(i);});

   transform(_src.begin(), _src.end(), std::back_inserter(src),
             [this](int i) { return create_src_register(i);});

   transform(_to.begin(), _to.end(), std::back_inserter(tex_offsets),
             [this](int i) { return create_src_register(i);});

}

MockCodeline::MockCodeline(unsigned _op, const vector<pair<int,int>>& _dst,
                           const vector<pair<int, const char *>>& _src,
                           const vector<pair<int, const char *>>&_to,
                           SWZ with_swizzle):
   op(_op),
   max_temp_id(0)
{
   (void)with_swizzle;

   transform(_dst.begin(), _dst.end(), std::back_inserter(dst),
             [this](pair<int,int> r) {
      return create_dst_register(r.first, r.second);
   });

   transform(_src.begin(), _src.end(), std::back_inserter(src),
             [this](const pair<int,const char *>& r) {
      return create_src_register(r.first, r.second);
   });

   transform(_to.begin(), _to.end(), std::back_inserter(tex_offsets),
             [this](const pair<int,const char *>& r) {
      return create_src_register(r.first, r.second);
   });
}

MockCodeline::MockCodeline(unsigned _op, const vector<tuple<int,int,int>>& _dst,
                           const vector<tuple<int,int,int>>& _src,
                           const vector<tuple<int,int,int>>&_to, RA with_reladdr):
   op(_op),
   max_temp_id(0)
{
   (void)with_reladdr;

   transform(_dst.begin(), _dst.end(), std::back_inserter(dst),
             [this](const tuple<int,int,int>& r) {
      return create_dst_register(r);
   });

   transform(_src.begin(), _src.end(), std::back_inserter(src),
             [this](const tuple<int,int,int>& r) {
      return create_src_register(r);
   });

   transform(_to.begin(), _to.end(), std::back_inserter(tex_offsets),
             [this](const tuple<int,int,int>& r) {
      return create_src_register(r);
   });
}

st_src_reg MockCodeline::create_src_register(int src_idx)
{
   return create_src_register(src_idx,
                              src_idx < 0 ? PROGRAM_INPUT : PROGRAM_TEMPORARY);
}

st_src_reg MockCodeline::create_src_register(int src_idx, const char *sw)
{
   st_src_reg result = create_src_register(src_idx);

   for (int i = 0; i < 4; ++i) {
      switch (sw[i]) {
      case 'x': break; /* is zero */
      case 'y': result.swizzle |= SWIZZLE_Y << 3 * i; break;
      case 'z': result.swizzle |= SWIZZLE_Z << 3 * i; break;
      case 'w': result.swizzle |= SWIZZLE_W << 3 * i; break;
      }
   }

   return result;
}

st_src_reg MockCodeline::create_src_register(int src_idx, gl_register_file file)
{
   st_src_reg retval;
   retval.file = file;
   retval.index = src_idx >= 0 ? src_idx  : 1 - src_idx;

   if (file == PROGRAM_TEMPORARY) {
      if (max_temp_id < src_idx)
         max_temp_id = src_idx;
   } else if (file == PROGRAM_ARRAY) {
      retval.array_id = 1;
   }
   retval.swizzle = SWIZZLE_XYZW;
   retval.type = GLSL_TYPE_INT;

   return retval;
}

st_src_reg *MockCodeline::create_rel_src_register(int idx)
{
   st_src_reg *retval = ralloc(mem_ctx, st_src_reg);
   *retval = st_src_reg(PROGRAM_TEMPORARY, idx, GLSL_TYPE_INT);
   if (max_temp_id < idx)
      max_temp_id = idx;
   return retval;
}

st_src_reg MockCodeline::create_src_register(const tuple<int,int,int>& src)
{
   int src_idx = std::get<0>(src);
   int relidx1 = std::get<1>(src);
   int relidx2 = std::get<2>(src);

   gl_register_file file = PROGRAM_TEMPORARY;
   if (src_idx < 0)
      file = PROGRAM_OUTPUT;
   else if (relidx1 || relidx2) {
      file = PROGRAM_ARRAY;
   }

   st_src_reg retval = create_src_register(src_idx, file);
   if (src_idx >= 0) {
      if (relidx1 || relidx2) {
         retval.array_id = 1;
         if (relidx1)
            retval.reladdr = create_rel_src_register(relidx1);
         if (relidx2) {
            retval.reladdr2 = create_rel_src_register(relidx2);
            retval.has_index2 = true;
            retval.index2D = 10;
         }
      }
   }
   return retval;
}

st_dst_reg MockCodeline::create_dst_register(int dst_idx,int writemask)
{
   gl_register_file file;
   int idx = 0;
   if (dst_idx >= 0) {
      file = PROGRAM_TEMPORARY;
      idx = dst_idx;
      if (max_temp_id < idx)
         max_temp_id = idx;
   } else {
      file = PROGRAM_OUTPUT;
      idx = 1 - dst_idx;
   }
   return st_dst_reg(file, writemask, GLSL_TYPE_INT, idx);
}

st_dst_reg MockCodeline::create_dst_register(int dst_idx)
{
   return create_dst_register(dst_idx, dst_idx < 0 ?
                                 PROGRAM_OUTPUT : PROGRAM_TEMPORARY);
}

st_dst_reg MockCodeline::create_dst_register(int dst_idx, gl_register_file file)
{
   st_dst_reg retval;
   retval.file = file;
   retval.index = dst_idx >= 0 ? dst_idx  : 1 - dst_idx;

   if (file == PROGRAM_TEMPORARY) {
      if (max_temp_id < dst_idx)
         max_temp_id = dst_idx;
   } else if (file == PROGRAM_ARRAY) {
      retval.array_id = 1;
   }
   retval.writemask = 0xF;
   retval.type = GLSL_TYPE_INT;

   return retval;
}

st_dst_reg MockCodeline::create_dst_register(const tuple<int,int,int>& dst)
{
   int dst_idx = std::get<0>(dst);
   int relidx1 = std::get<1>(dst);
   int relidx2 = std::get<2>(dst);

   gl_register_file file = PROGRAM_TEMPORARY;
   if (dst_idx < 0)
      file = PROGRAM_OUTPUT;
   else if (relidx1 || relidx2) {
      file = PROGRAM_ARRAY;
   }
   st_dst_reg retval = create_dst_register(dst_idx, file);

   if (relidx1 || relidx2) {
      if (relidx1)
         retval.reladdr = create_rel_src_register(relidx1);
      if (relidx2) {
         retval.reladdr2 = create_rel_src_register(relidx2);
         retval.has_index2 = true;
         retval.index2D = 10;
      }
   }
   return retval;
}

glsl_to_tgsi_instruction *MockCodeline::get_codeline() const
{
   glsl_to_tgsi_instruction *next_instr = new(mem_ctx) glsl_to_tgsi_instruction();
   next_instr->op = op;
   next_instr->info = tgsi_get_opcode_info(op);

   assert(src.size() == num_inst_src_regs(next_instr));
   assert(dst.size() == num_inst_dst_regs(next_instr));
   assert(tex_offsets.size() < 3);

   copy(src.begin(), src.end(), next_instr->src);
   copy(dst.begin(), dst.end(), next_instr->dst);

   next_instr->tex_offset_num_offset = tex_offsets.size();

   if (next_instr->tex_offset_num_offset > 0) {
      next_instr->tex_offsets = ralloc_array(mem_ctx, st_src_reg, tex_offsets.size());
      copy(tex_offsets.begin(), tex_offsets.end(), next_instr->tex_offsets);
   } else {
      next_instr->tex_offsets = nullptr;
   }
   return next_instr;
}

void MockCodeline::set_mem_ctx(void *ctx)
{
   mem_ctx = ctx;
}


MockShader::MockShader(const vector<MockCodeline>& source, void *ctx):
   num_temps(0)
{
   program = new(ctx) exec_list();

   for (const MockCodeline& i: source) {
      program->push_tail(i.get_codeline());
      int t = i.get_max_reg_id();
      if (t > num_temps)
         num_temps = t;
   }

   ++num_temps;
}

int MockShader::get_num_temps() const
{
   return num_temps;
}

exec_list* MockShader::get_program() const
{
   return program;
}

void MesaTestWithMemCtx::SetUp()
{
   mem_ctx = ralloc_context(nullptr);
   MockCodeline::set_mem_ctx(mem_ctx);
}

void MesaTestWithMemCtx::TearDown()
{
   ralloc_free(mem_ctx);
   MockCodeline::set_mem_ctx(nullptr);
   mem_ctx = nullptr;
}

void LifetimeEvaluatorTest::run(const vector<MockCodeline>& code, const expectation& e)
{
   MockShader shader(code, mem_ctx);
   std::vector<lifetime> result(shader.get_num_temps());

   bool success =
         get_temp_registers_required_lifetimes(mem_ctx, shader.get_program(),
                                               shader.get_num_temps(), &result[0]);

   ASSERT_TRUE(success);
   ASSERT_EQ(result.size(), e.size());
   check(result, e);
}

void LifetimeEvaluatorExactTest::check( const vector<lifetime>& lifetimes,
                                        const expectation& e)
{
   for (unsigned i = 1; i < lifetimes.size(); ++i) {
      EXPECT_EQ(lifetimes[i].begin, e[i][0]);
      EXPECT_EQ(lifetimes[i].end, e[i][1]);
   }
}

void LifetimeEvaluatorAtLeastTest::check( const vector<lifetime>& lifetimes,
                                          const expectation& e)
{
   for (unsigned i = 1; i < lifetimes.size(); ++i) {
      EXPECT_LE(lifetimes[i].begin, e[i][0]);
      EXPECT_GE(lifetimes[i].end, e[i][1]);
   }
}

void RegisterRemappingTest::run(const vector<lifetime>& lt,
                            const vector<int>& expect)
{
   rename_reg_pair proto{false,0};
   vector<rename_reg_pair> result(lt.size(), proto);

   get_temp_registers_remapping(mem_ctx, lt.size(), &lt[0], &result[0]);

   vector<int> remap(lt.size());
   for (unsigned i = 0; i < lt.size(); ++i) {
      remap[i] = result[i].valid ? result[i].new_reg : i;
   }

   std::transform(remap.begin(), remap.end(), result.begin(), remap.begin(),
                  [](int x, const rename_reg_pair& rn) {
                     return rn.valid ? rn.new_reg : x;
                  });

   for(unsigned i = 1; i < remap.size(); ++i) {
      EXPECT_EQ(remap[i], expect[i]);
   }
}

void RegisterLifetimeAndRemappingTest::run(const vector<MockCodeline>& code,
                                           const vector<int>& expect)
{
     MockShader shader(code, mem_ctx);
     std::vector<lifetime> lt(shader.get_num_temps());
     get_temp_registers_required_lifetimes(mem_ctx, shader.get_program(),
                                           shader.get_num_temps(), &lt[0]);
     this->run(lt, expect);
}