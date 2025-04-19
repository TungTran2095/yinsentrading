import { configureStore } from '@reduxjs/toolkit';
import botsReducer from './botsSlice';
import backtestReducer from './backtestSlice';
import modelsReducer from './modelsSlice';

const store = configureStore({
  reducer: {
    bots: botsReducer,
    backtest: backtestReducer,
    models: modelsReducer,
  },
});

export default store;
