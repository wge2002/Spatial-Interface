"""Check context retention without dropping tool history or changing source logs."""
import copy
import asyncio
from pathlib import Path
import sys
import tempfile
import unittest

try:
    ExceptionGroup
except NameError:  # DSW Python 3.10 uses AnyIO's backport.
    from exceptiongroup import ExceptionGroup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial_interface.qwen_agent import Agent, ContextBudgetError, bound_image_history, exception_details


class ContextHistory(unittest.TestCase):
    def test_old_images_removed_but_recent_bytes_and_tool_pairs_preserved(self):
        messages = [{"role": "system", "content": "guide"}, {"role": "user", "content": "goal"}]
        for i in range(4):
            messages += [
                {"role": "assistant", "tool_calls": [{"id": str(i)}], "content": "plan"},
                {"role": "tool", "tool_call_id": str(i), "content": "metric evidence"},
                {"role": "user", "content": [{"type": "text", "text": f"source {i}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{i}a"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{i}b"}}]}]
        original = copy.deepcopy(messages)
        bounded, removed = bound_image_history(messages, 2)
        self.assertEqual(removed, 4)
        self.assertEqual(messages, original)
        self.assertEqual(bounded[-6:], messages[-6:])
        self.assertEqual([m for m in bounded if m['role'] in ('assistant', 'tool')],
                         [m for m in messages if m['role'] in ('assistant', 'tool')])
        self.assertEqual(bound_image_history(bounded, 2), (bounded, 0))
        self.assertIn('omitted', bounded[4]['content'][1]['text'])

    def test_default_keeps_historical_context_and_invalid_limits_fail(self):
        messages = [{"role": "user", "content": "prompt"}]
        self.assertIs(bound_image_history(messages, None)[0], messages)
        for invalid in (0, -1, True, "2", 1.5):
            with self.assertRaises(ValueError):
                bound_image_history(messages, invalid)

    def test_nested_error_retains_actionable_cause(self):
        error = ExceptionGroup('TaskGroup', [ExceptionGroup('nested', [RuntimeError('HTTP 400: context limit')])])
        self.assertEqual(exception_details(error), 'RuntimeError: HTTP 400: context limit')

    def test_context_400_is_classified_and_logged_without_retry(self):
        class Response:
            status_code=400
            async def __aenter__(self): return self
            async def __aexit__(self,*args): pass
            async def aread(self):
                return b'{"message":"Requested token count exceeds the maximum context length"}'
        class Client:
            count=0
            def stream(self,*args,**kwargs):
                self.count+=1
                return Response()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'key').write_text('unit-test-key')
            agent=Agent(dict(output=str(root/'out'),timeout=30,model='fake',seed=1,mode='direct_geometry',
                server=dict(key_file=str(root/'key'),base_url='http://unused.invalid/v1'),sampling={}))
            client=Client()
            try:
                with self.assertRaises(ContextBudgetError):
                    asyncio.run(agent.request(client,[{'role':'user','content':'goal'}],[]))
                self.assertEqual(client.count,1)
                self.assertIn('ContextBudgetError',agent.requests[0]['error'])
                self.assertIsNone(agent.requests[0]['usage'])
                self.assertTrue((root/'out/request_001/timing.json').exists())
            finally:
                agent.events.close()


if __name__ == '__main__':
    unittest.main()
